# 6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyTrend
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and 1w trend confirmation.
# Long when price breaks above R3 with price above 1d EMA34 and above 1w EMA34.
# Short when price breaks below S3 with price below 1d EMA34 and below 1w EMA34.
# Exit when price returns to Camarilla pivot level (mean reversion within range).
# Uses daily and weekly trends to filter breakouts, working in both bull and bear markets.
# Target: 20-50 trades/year to minimize fee drag.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

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

    # Calculate Camarilla levels from previous day
    # Need previous day's high, low, close
    # Since we're on 6h timeframe, we need to group by day
    # We'll calculate daily OHLC first
    
    # Create date column for grouping
    dates = pd.to_datetime(prices['open_time']).dt.date
    
    # Calculate daily OHLC
    daily_high = pd.Series(high).groupby(dates).transform('max')
    daily_low = pd.Series(low).groupby(dates).transform('min')
    daily_close = pd.Series(close).groupby(dates).transform('last')
    
    # Previous day's values (shift by 1)
    prev_high = daily_high.shift(1)
    prev_low = daily_low.shift(1)
    prev_close = daily_close.shift(1)
    
    # Calculate Camarilla levels
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    range_hl = prev_high - prev_low
    camarilla_pivot = prev_close
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    
    # Get 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 1w EMA34 for trend confirmation (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if data is not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R3 with price above 1d EMA34 and 1w EMA34 (uptrend)
            if close[i] > r3[i] and close[i] > ema_1d_aligned[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with price below 1d EMA34 and 1w EMA34 (downtrend)
            elif close[i] < s3[i] and close[i] < ema_1d_aligned[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot level (mean reversion)
            if close[i] <= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot level
            if close[i] >= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals