#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation.
    # Uses 1d Camarilla levels for precise support/resistance, 1w EMA for trend direction,
    # and 1d volume spike for participation confirmation. Designed for low trade frequency
    # (target: 30-100 total over 4 years) to minimize fee drag in both bull and bear markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using standard multipliers)
    camarilla_h4 = pivot + (range_1d * 1.1 / 2)
    camarilla_h3 = pivot + (range_1d * 1.1 / 4)
    camarilla_h2 = pivot + (range_1d * 1.1 / 6)
    camarilla_l2 = pivot - (range_1d * 1.1 / 6)
    camarilla_l3 = pivot - (range_1d * 1.1 / 4)
    camarilla_l4 = pivot - (range_1d * 1.1 / 2)
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d timeframe (prices is already 1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma_aligned[i]
        
        # Trend filter: price above/below 1w EMA(21)
        uptrend = close[i] > ema_21_aligned[i]
        downtrend = close[i] < ema_21_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = close[i] > camarilla_h4_aligned[i]  # Break above H4
        breakout_short = close[i] < camarilla_l4_aligned[i]  # Break below L4
        
        # Entry conditions: breakout in direction of trend with volume confirmation
        long_entry = breakout_long and uptrend and volume_filter
        short_entry = breakout_short and downtrend and volume_filter
        
        # Exit conditions: price returns to opposite Camarilla level
        long_exit = close[i] < camarilla_l4_aligned[i]
        short_exit = close[i] > camarilla_h4_aligned[i]
        
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

name = "1d_1w_camarilla_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0