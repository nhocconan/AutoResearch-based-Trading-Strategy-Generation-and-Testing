#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend(10,3) + 1d VWAP mean reversion with volume confirmation
# Long when: price < 1d VWAP - 1.5*ATR(1d) AND 6h Supertrend = uptrend AND volume > 1.5*20-period avg volume
# Short when: price > 1d VWAP + 1.5*ATR(1d) AND 6h Supertrend = downtrend AND volume > 1.5*20-period avg volume
# Exit when price crosses 1d VWAP (mean reversion complete) or Supertrend flips
# Uses discrete sizing 0.25 to manage drawdown (BTC -77% in 2022 → ~19.3% loss at 0.25 exposure)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Combines short-term mean reversion (VWAP bands) with medium-term trend filter (Supertrend) and volume confirmation
# VWAP acts as dynamic support/resistance, Supertrend filters counter-trend trades, volume ensures conviction

name = "6h_Supertrend_1dVWAP_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for VWAP and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (typical price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (np.cumsum(typical_price_1d * volume_1d) / np.cumsum(volume_1d))
    
    # Calculate 1d ATR(14) for VWAP bands
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h Supertrend(10,3)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_6h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + (3.0 * atr_6h)
    lower_basic = hl2 - (3.0 * atr_6h)
    
    # Final Upper and Lower Bands
    upper_band = np.zeros_like(close)
    lower_band = np.zeros_like(close)
    upper_band[0] = upper_basic[0]
    lower_band[0] = lower_basic[0]
    
    for i in range(1, len(close)):
        # Upper Band
        if upper_basic[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i-1]
        
        # Lower Band
        if lower_basic[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # VWAP bands for mean reversion
        vwap_upper = vwap_1d_aligned[i] + (1.5 * atr_1d_aligned[i])
        vwap_lower = vwap_1d_aligned[i] - (1.5 * atr_1d_aligned[i])
        
        if position == 0:
            # Mean reversion signals with trend and volume filters
            # Long: price below lower VWAP band AND uptrend AND volume spike
            if close[i] < vwap_lower and direction[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above upper VWAP band AND downtrend AND volume spike
            elif close[i] > vwap_upper and direction[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above VWAP (mean reversion complete) OR trend flips
            if close[i] > vwap_1d_aligned[i] or direction[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below VWAP (mean reversion complete) OR trend flips
            if close[i] < vwap_1d_aligned[i] or direction[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals