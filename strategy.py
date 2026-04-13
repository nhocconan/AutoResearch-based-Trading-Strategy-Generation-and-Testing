# 4h_1d_camarilla_breakout_vol_filter
# Hypothesis: Camarilla pivot levels on 1d act as strong support/resistance in both bull and bear markets.
# Price touching L3 (bullish) or H3 (bearish) with volume confirmation provides high-probability entries.
# Volume filter reduces false breakouts. Trend filter (price vs 200-period EMA) avoids counter-trend trades.
# Target: 20-40 trades/year per symbol with ~60% win rate. Works in trends (breakouts) and ranges (mean reversion at extremes).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using previous day's range)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val <= 0:
            continue
            
        camarilla_h3[i] = prev_close + 1.1 * range_val / 6
        camarilla_l3[i] = prev_close - 1.1 * range_val / 6
        camarilla_h4[i] = prev_close + 1.1 * range_val / 2
        camarilla_l4[i] = prev_close - 1.1 * range_val / 2
    
    # Calculate 200-period EMA on 1d for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 20-period volume average for volume filter
    volume_20 = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        volume_20[i] = np.mean(volume[i-20:i])
    
    # Align indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    volume_20_aligned = align_htf_to_ltf(prices, df_1d, volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period average
        vol_filter = volume[i] > volume_20_aligned[i]
        
        # Trend filter: price above/below 200 EMA
        above_ema = close[i] > ema_200_aligned[i]
        below_ema = close[i] < ema_200_aligned[i]
        
        # Camarilla touch conditions
        touch_h3 = high[i] >= h3_aligned[i] and low[i] <= h3_aligned[i]  # price touched H3
        touch_l3 = high[i] >= l3_aligned[i] and low[i] <= l3_aligned[i]  # price touched L3
        
        # Entry conditions
        long_entry = touch_l3 and above_ema and vol_filter  # bullish touch in uptrend
        short_entry = touch_h3 and below_ema and vol_filter  # bearish touch in downtrend
        
        # Exit conditions: opposite touch or mean reversion to center
        exit_long = position == 1 and (touch_h3 or close[i] >= (h3_aligned[i] + l3_aligned[i]) / 2)
        exit_short = position == -1 and (touch_l3 or close[i] <= (h3_aligned[i] + l3_aligned[i]) / 2)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_filter"
timeframe = "4h"
leverage = 1.0