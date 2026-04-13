#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: Daily Camarilla pivot breakout with volume confirmation and weekly trend filter.
    # Long when price breaks above H3 with volume spike and weekly close > weekly open.
    # Short when price breaks below L3 with volume spike and weekly close < weekly open.
    # Exit when price returns to daily pivot point (mean reversion).
    # Uses 1d Camarilla levels and 1w trend aligned to daily bars. Discrete size 0.25.
    # Target: 30-100 total trades over 4 years (7-25/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivots (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Close + Range * 1.1 / 4
    # L3 = Close - Range * 1.1 / 4
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    
    # H3 and L3 are the key breakout levels
    camarilla_h3 = close_1d + rng * 1.1 / 4.0
    camarilla_l3 = close_1d - rng * 1.1 / 4.0
    camarilla_pivot = pivot  # Exit level
    
    # Calculate 1d volume mean (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open, bearish if weekly close < weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align HTF indicators to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume filter: current 1d volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Entry conditions: price breaks Camarilla H3/L3 levels with volume confirmation and weekly trend
        long_entry = (close[i] > camarilla_h3_aligned[i] and 
                      volume_confirmation and 
                      weekly_bullish_aligned[i] > 0.5)
        short_entry = (close[i] < camarilla_l3_aligned[i] and 
                       volume_confirmation and 
                       weekly_bearish_aligned[i] > 0.5)
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion)
        long_exit = close[i] < camarilla_pivot_aligned[i]
        short_exit = close[i] > camarilla_pivot_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0