#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h pivot-based breakouts with volume and momentum confirmation work in both bull and bear regimes.
# Uses 1d pivots (R1/S1) as institutional levels. Volume surge confirms breakout strength.
# RSI(14) filters overbought/oversold conditions to avoid false breakouts.
# ATR-based stop-loss manages risk. Target: 15-25 trades/year per symbol (<100 total over 4 years).
# Timeframe: 12h balances signal quality and trade frequency to minimize fee drag.
name = "12h_Pivot_R1S1_Breakout_VolumeRSI_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point and R1/S1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Momentum filter: RSI(14) to avoid overextended entries
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # ATR-based stop-loss calculation (14-period)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        rsi_val = rsi_values[i]
        atr = atr_14[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: break above R1 with volume and RSI not overbought
            if price > r1 and volume_confirmed and rsi_val < 70:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and RSI not oversold
            elif price < s1 and volume_confirmed and rsi_val > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below R1 or RSI overbought
            if price < r1 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above S1 or RSI oversold
            if price > s1 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals