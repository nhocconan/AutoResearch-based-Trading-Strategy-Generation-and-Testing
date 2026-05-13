# 6h_RangeBreakout_WeeklyTrend
# Hypothesis: In ranging markets (identified by weekly low volatility), 
# breakouts from daily consolidation ranges (identified by narrow daily range) 
# with volume confirmation capture explosive moves. Works in both bull and bear 
# by trading breakouts in the direction of the weekly trend. 
# Target: 15-25 trades/year per symbol.

name = "6h_RangeBreakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for consolidation and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Daily range (high - low) as percentage of price
    daily_range_pct = (high_1d - low_1d) / close_1d
    # 20-period average of daily range percentage
    avg_daily_range_pct = pd.Series(daily_range_pct).rolling(window=20, min_periods=20).mean().values
    # Current daily range as percentage
    current_daily_range_pct = (high_1d - low_1d) / close_1d

    # Daily consolidation: today's range < 50% of 20-day average range
    daily_consolidation = current_daily_range_pct < (0.5 * avg_daily_range_pct)

    # Weekly trend: 50-period EMA on weekly close
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > weekly_ema50
    weekly_downtrend = close_1w < weekly_ema50

    # 6h consolidation range: highest high and lowest low of last 4 bars (24h)
    highest_high_24h = np.maximum.accumulate(high[::-1])[::-1]
    lowest_low_24h = np.minimum.accumulate(low[::-1])[::-1]
    # Shift to get past 4 bars only
    highest_high_past = np.roll(highest_high_24h, 4)
    lowest_low_past = np.roll(lowest_low_24h, 4)
    # For first 4 bars, use NaN
    highest_high_past[:4] = np.nan
    lowest_low_past[:4] = np.nan

    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup for daily averages
        # Get aligned daily values for current 6h bar
        if i < len(daily_consolidation):
            cons = daily_consolidation[i]
        else:
            cons = False
            
        if i < len(weekly_uptrend):
            up_trend = weekly_uptrend[i]
            down_trend = weekly_downtrend[i]
        else:
            up_trend = False
            down_trend = False
            
        if i < len(volume_spike):
            vol_spike = volume_spike[i]
        else:
            vol_spike = False

        # Handle NaN values from alignment
        if (np.isnan(highest_high_past[i]) or np.isnan(lowest_low_past[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend + daily consolidation + upside breakout + volume spike
            if (up_trend and cons and 
                close[i] > highest_high_past[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + daily consolidation + downside breakout + volume spike
            elif (down_trend and cons and 
                  close[i] < lowest_low_past[i] and vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 24h low or weekly trend changes
            if (close[i] < lowest_low_past[i] or not up_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 24h high or weekly trend changes
            if (close[i] > highest_high_past[i] or not down_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals