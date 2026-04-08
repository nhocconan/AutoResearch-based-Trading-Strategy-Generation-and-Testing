#!/usr/bin/env python3
# 4h_1w_donchian_breakout_volume_filter_v2
# Hypothesis: Breakout of 4-hour Donchian channel (20-period) with weekly volume confirmation (>1.5x 4-week average) and trend filter (price above/below 200-period EMA).
# Long when price breaks above upper Donchian band with weekly volume surge and price > EMA200.
# Short when price breaks below lower Donchian band with weekly volume surge and price < EMA200.
# Uses weekly volume to filter false breakouts and EMA200 to avoid counter-trend trades.
# Designed for 20-30 trades/year on 4h to minimize fee decay while capturing strong momentum moves.
# Works in bull markets via upside breakouts and bear markets via downside breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_donchian_breakout_volume_filter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4-hour EMA(200) for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1-week volume data for confirmation
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # 1-week volume moving average (4-period) for surge detection
    vol_ma_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_4_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure EMA(200) is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema200[i]) or np.isnan(vol_ma_4_1w_aligned[i]) or np.isnan(volume_1w_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current 1w volume > 1.5x 4-period average
        vol_surge = volume_1w_aligned[i] > 1.5 * vol_ma_4_1w_aligned[i] if vol_ma_4_1w_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or trend filter fails
            if close[i] < low_roll[i] or close[i] < ema200[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or trend filter fails
            if close[i] > high_roll[i] or close[i] > ema200[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band with volume surge and uptrend
            if close[i] > high_roll[i] and vol_surge and close[i] > ema200[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band with volume surge and downtrend
            elif close[i] < low_roll[i] and vol_surge and close[i] < ema200[i]:
                position = -1
                signals[i] = -0.25
    
    return signals