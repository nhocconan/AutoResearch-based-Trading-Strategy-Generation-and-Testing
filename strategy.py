#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot S1/S2 breakout with 1d volume confirmation and ATR stoploss
# Camarilla pivots identify intraday support/resistance levels; S1/S2 breakouts capture trend continuation
# 1d volume > 1.8x 20-period average confirms institutional participation on higher timeframe
# ATR-based stop loss manages risk; designed for 12h timeframe with selective entries to avoid overtrading
# Target: 15-35 trades per year per symbol (60-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and volume filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for S1, S2, R1, R2
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    hl_range = high_1d - low_1d
    r1 = close_1d + (hl_range * 1.0833)
    r2 = close_1d + (hl_range * 1.1666)
    s1 = close_1d - (hl_range * 1.0833)
    s2 = close_1d - (hl_range * 1.1666)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: 1d volume > 1.8x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = volume_1d > (vol_ma_1d * 1.8)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # Calculate ATR for stop loss (using 12h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or \
           np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_filter_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        has_volume = vol_filter_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + volume confirmation
            long_signal = (price > r1_aligned[i]) and has_volume
            
            # Short entry: price breaks below S1 + volume confirmation
            short_signal = (price < s1_aligned[i]) and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or S2 break (mean reversion)
            stop_loss = entry_price - 2.0 * atr[i]
            s2_break = price < s2_aligned[i]
            
            if stop_loss <= 0 or price <= stop_loss or s2_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or R2 break (mean reversion)
            stop_loss = entry_price + 2.0 * atr[i]
            r2_break = price > r2_aligned[i]
            
            if stop_loss <= 0 or price >= stop_loss or r2_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_S1S2_Breakout_Volume_ATR"
timeframe = "12h"
leverage = 1.0