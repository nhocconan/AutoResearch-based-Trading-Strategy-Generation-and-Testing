# 12h_Ichimoku_Cloud_TenkanKijun_Cross_1dTrend
# Ichimoku Cloud system: Tenkan-Kijun cross with cloud color filter + 1d trend alignment
# Works in both bull and bear: buys when Tenkan crosses above Kijun above cloud in uptrend,
# sells when Tenkan crosses below Kijun below cloud in downtrend. Uses volume confirmation.

name = "12h_Ichimoku_Cloud_TenkanKijun_Cross_1dTrend"
timeframe = "12h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)

    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)

    # Cloud color: green when Senkou A > Senkou B (bullish), red when A < B (bearish)
    # We'll use the current cloud (not shifted) for simplicity in filtering
    # For cloud color at time i, we need Senkou A/B from 26 periods ago
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    cloud_green = senkou_a_lag > senkou_b_lag  # True when cloud is bullish

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Need enough data for Ichimoku
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lag[i]) or np.isnan(senkou_b_lag[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Cloud boundaries at current time (using lagged Senkou lines)
        upper_cloud = np.maximum(senkou_a_lag[i], senkou_b_lag[i])
        lower_cloud = np.minimum(senkou_a_lag[i], senkou_b_lag[i])

        if position == 0:
            # LONG: Tenkan crosses above Kijun AND price above cloud AND bullish cloud AND 1d uptrend + volume
            if (tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i] and  # Cross
                close[i] > upper_cloud and                           # Above cloud
                cloud_green[i] and                                   # Bullish cloud
                close[i] > ema34_1d_aligned[i] and                   # 1d uptrend
                volume[i] > vol_avg_20[i] * 1.5):                    # Volume spike
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun AND price below cloud AND bearish cloud AND 1d downtrend + volume
            elif (tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i] and  # Cross
                  close[i] < lower_cloud and                            # Below cloud
                  not cloud_green[i] and                                # Bearish cloud
                  close[i] < ema34_1d_aligned[i] and                    # 1d downtrend
                  volume[i] > vol_avg_20[i] * 1.5):                     # Volume spike
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price breaks below cloud OR 1d trend turns down
            if (tenkan[i] < kijun[i] or 
                close[i] < lower_cloud or
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price breaks above cloud OR 1d trend turns up
            if (tenkan[i] > kijun[i] or 
                close[i] > upper_cloud or
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals