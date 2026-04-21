#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1
Hypothesis: Daily Camarilla pivot R1/S1 breakout with volume spike (>1.5x 20-day volume MA) and ATR-based stoploss. Uses 1w HTF trend filter (price > EMA34 weekly for longs, < EMA34 weekly for shorts). Designed for very low trade frequency (~50-80 total over 4 years) to minimize fee drag and work in both bull/bear markets via weekly trend alignment. Focus on BTC/ETH as primary targets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume MA (20-day) for spike detection
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-day) for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if indicators not ready
        if (np.isnan(vol_ma[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need at least 2 days of data for Camarilla calculation (yesterday's OHLC)
        if i < 1:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla pivot levels based on PREVIOUS day's OHLC
        # (lookback by 1 to avoid look-ahead)
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Camarilla R1 and S1 levels
        r1 = pivot + (range_hl * 1.1 / 12.0)
        s1 = pivot - (range_hl * 1.1 / 12.0)
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Weekly trend filter
        weekly_uptrend = price > ema_34_1w_aligned[i]
        weekly_downtrend = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and weekly uptrend
            if price > r1 and vol_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume spike and weekly downtrend
            elif price < s1 and vol_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Camarilla H3-L3 range (mean reversion exit)
            elif price < pivot + (range_hl * 1.1 / 6.0) and price > pivot - (range_hl * 1.1 / 6.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Camarilla H3-L3 range
            elif price < pivot + (range_hl * 1.1 / 6.0) and price > pivot - (range_hl * 1.1 / 6.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1"
timeframe = "1d"
leverage = 1.0