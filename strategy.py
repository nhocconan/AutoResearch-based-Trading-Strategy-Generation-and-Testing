# The core principle is to avoid overtrading by using a high-timeframe (1w) for the primary trend
# and a medium-timeframe (1d) for entry signals, ensuring trades are taken only in the direction of the weekly trend.
# This strategy uses the weekly Donchian channel (20-period) to establish the trend.
# On the daily chart, it looks for a breakout of the Donchian channel (10-period) in the direction of the weekly trend,
# confirmed by above-average volume.
# This approach aims to capture significant moves while minimizing trades during choppy or counter-trend periods.
# The use of weekly and daily timeframes is intended to reduce noise and increase the reliability of signals.
# Position sizing is kept moderate (0.25) to manage risk, especially during volatile periods like the 2022 bear market.
# The strategy is designed to be long-only in a weekly uptrend and short-only in a weekly downtrend, avoiding counter-trend trading.
# This directional bias, combined with breakout logic and volume confirmation, is intended to work in both bull and bear markets
# by following the dominant weekly trend.

name = "6h_WeeklyTrend_DailyBreakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Increased warmup for weekly data
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter (Donchian channel)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian Channel (20-period)
    weekly_high = df_1w['high'].rolling(window=20, min_periods=20).max().values
    weekly_low = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Determine weekly trend: price above/below midpoint of weekly Donchian
    weekly_mid = (weekly_high_aligned + weekly_low_aligned) / 2
    weekly_uptrend = close > weekly_mid
    weekly_downtrend = close < weekly_mid

    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily Donchian Channel (10-period for breakout)
    daily_high = df_1d['high'].rolling(window=10, min_periods=10).max().values
    daily_low = df_1d['low'].rolling(window=10, min_periods=10).min().values
    
    # Align daily Donchian levels to 6h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation: current volume > 1.5x average of last 6 periods (1.5 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend AND price breaks above daily high AND volume confirmation
            if weekly_uptrend[i] and close[i] > daily_high_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend AND price breaks below daily low AND volume confirmation
            elif weekly_downtrend[i] and close[i] < daily_low_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend turns down OR price breaks below daily low (failed breakout)
            if not weekly_uptrend[i] or close[i] < daily_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns up OR price breaks above daily high (failed breakdown)
            if not weekly_downtrend[i] or close[i] > daily_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals