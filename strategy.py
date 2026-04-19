#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and daily volatility filter.
# Long when price breaks above R1 (Camarilla resistance) with volume > 1.5x daily average volume and daily ATR(14) < daily ATR(50) (low volatility regime)
# Short when price breaks below S1 (Camarilla support) with volume > 1.5x daily average volume and daily ATR(14) < daily ATR(50)
# Exit when price crosses back through the Camarilla pivot point (P).
# Uses Camarilla pivot levels for structured support/resistance, volume for conviction, daily volatility filter to avoid chop.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_VolRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation, volume average, and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # P = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # We use the previous day's values to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate daily ATR for volatility regime filter
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - np.roll(prev_close, 1))
    tr3 = np.abs(prev_low - np.roll(prev_close, 1))
    tr2[0] = np.inf  # First value has no previous close
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily average volume for confirmation
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily arrays to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        vol = volume[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Volatility regime filter: only trade in low volatility (ATR14 < ATR50)
        vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: break above R1 + volume spike + low vol regime
            if price > r1_val and vol > 1.5 * vol_ma_val and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + low vol regime
            elif price < s1_val and vol > 1.5 * vol_ma_val and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals