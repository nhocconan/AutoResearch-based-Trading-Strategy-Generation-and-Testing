#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend direction and 1d for volume spike filter.
- Donchian(20): Break above 20-period high for long, below 20-period low for short.
- Trend Filter: Price > EMA50(1w) for long bias, Price < EMA50(1w) for short bias.
- Volume Confirmation: Current 12h volume > 2.0 * 20-period average 1d volume (scaled to 12h).
- Entry: Long when price > Donchian high(20) AND price > EMA50(1w) AND volume confirmation.
         Short when price < Donchian low(20) AND price < EMA50(1w) AND volume confirmation.
- Exit: Opposite Donchian breakout (long exits when price < Donchian low(10), short exits when price > Donchian high(10)).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume average for confirmation (20-period) - scaled to 12h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    # Scale 1d average volume to 12h: 12h = 0.5 * 1d (since 12h is half a day)
    vol_ma_20_12h_scaled = vol_ma_20_1d_aligned * 0.5
    
    # Calculate Donchian channels (20 for entry, 10 for exit) on 12h timeframe
    lookback_entry = 20
    lookback_exit = 10
    
    highest_high_20 = pd.Series(high).rolling(window=lookback_entry, min_periods=lookback_entry).max().values
    lowest_low_20 = pd.Series(low).rolling(window=lookback_entry, min_periods=lookback_entry).min().values
    highest_high_10 = pd.Series(high).rolling(window=lookback_exit, min_periods=lookback_exit).max().values
    lowest_low_10 = pd.Series(low).rolling(window=lookback_exit, min_periods=lookback_exit).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_entry, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_12h_scaled[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(highest_high_10[i]) or np.isnan(lowest_low_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50(1w) for long bias, price < EMA50(1w) for short bias
        long_bias = curr_close > ema50_1w_aligned[i]
        short_bias = curr_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period average 12h volume (scaled from 1d)
        volume_confirm = curr_volume > 2.0 * vol_ma_20_12h_scaled[i]
        
        # Donchian breakout conditions
        breakout_high = curr_high > highest_high_20[i-1] if i > 0 else False  # Use previous bar's channel
        breakout_low = curr_low < lowest_low_20[i-1] if i > 0 else False
        
        # Donchian exit conditions (using 10-period channels)
        exit_low = curr_low < lowest_low_10[i]   # Exit long when price breaks below 10-period low
        exit_high = curr_high > highest_high_10[i]  # Exit short when price breaks above 10-period high
        
        # Exit conditions: opposite Donchian breakout (using shorter period for smoother exit)
        if position != 0:
            # Exit long: price breaks below 10-period Donchian low
            if position == 1:
                if exit_low:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above 10-period Donchian high
            elif position == -1:
                if exit_high:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above 20-period Donchian high AND long bias AND volume confirmation
            long_condition = breakout_high and long_bias and volume_confirm
            
            # Short: break below 20-period Donchian low AND short bias AND volume confirmation
            short_condition = breakout_low and short_bias and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50Trend_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0