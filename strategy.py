#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1w close > EMA50 (uptrend) AND volume spike.
# Short when price breaks below Donchian(20) low AND 1w close < EMA50 (downtrend) AND volume spike.
# Uses Donchian channel for breakout signals, 1-week EMA50 for trend direction, and volume to confirm momentum.
# Designed for low trade frequency (target: 20-50/year) to minimize fee drag and improve robustness.
# Works in bull markets via breakout longs in uptrend and in bear markets via breakout shorts in downtrend.
name = "1d_DonchianBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 1d data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Breakout signals: above upper band (long), below lower band (short)
    breakout_long = close > highest_high_20
    breakout_short = close < lowest_low_20
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w trend: close > EMA50 (uptrend), close < EMA50 (downtrend)
    trend_up = close_1w > ema_50_1w
    trend_down = close_1w < ema_50_1w
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + 1w uptrend + volume spike
            long_condition = breakout_long[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: Donchian breakout down + 1w downtrend + volume spike
            short_condition = breakout_short[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Donchian mid-line or 1w trend turns down
            mid_line = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if close[i] < mid_line or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Donchian mid-line or 1w trend turns up
            mid_line = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if close[i] > mid_line or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals