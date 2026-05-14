#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Trend_Regime
# Hypothesis: TRIX (15-period) detects momentum changes; long when TRIX turns up with volume spike and daily uptrend, short when TRIX turns down with volume spike and daily downtrend.
# Uses 1-day EMA50 for trend filter and volume > 2x 20-period average for confirmation. Designed for 4h timeframe to avoid overtrading.
# Works in bull markets via momentum continuation and in bear markets via mean-reversion bounces at key levels.

name = "4h_TRIX_VolumeSpike_Trend_Regime"
timeframe = "4h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # TRIX (15-period) on close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (pd.Series(ema3).pct_change()).values

    # Volume confirmation: current volume > 2x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: TRIX turning up (positive slope) with volume spike and daily uptrend
            if i > 0 and not np.isnan(trix[i-1]) and trix[i] > trix[i-1] and volume_ok[i] and price_above_daily_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turning down (negative slope) with volume spike and daily downtrend
            elif i > 0 and not np.isnan(trix[i-1]) and trix[i] < trix[i-1] and volume_ok[i] and price_below_daily_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns down or trend turns down
            if i > 0 and not np.isnan(trix[i-1]) and trix[i] < trix[i-1] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns up or trend turns up
            if i > 0 and not np.isnan(trix[i-1]) and trix[i] > trix[i-1] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
  #!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Trend_Regime
# Hypothesis: TRIX (15-period) detects momentum changes; long when TRIX turns up with volume spike and daily uptrend, short when TRIX turns down with volume spike and daily downtrend.
# Uses 1-day EMA50 for trend filter and volume > 2x 20-period average for confirmation. Designed for 4h timeframe to avoid overtrading.
# Works in bull markets via momentum continuation and in bear markets via mean-reversion bounces at key levels.

name = "4h_TRIX_VolumeSpike_Trend_Regime"
timeframe = "4h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # TRIX (15-period) on close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (pd.Series(ema3).pct_change()).values

    # Volume confirmation: current volume > 2x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: TRIX turning up (positive slope) with volume spike and daily uptrend
            if i > 0 and not np.isnan(trix[i-1]) and trix[i] > trix[i-1] and volume_ok[i] and price_above_daily_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turning down (negative slope) with volume spike and daily downtrend
            elif i > 0 and not np.isnan(trix[i-1]) and trix[i] < trix[i-1] and volume_ok[i] and price_below_daily_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns down or trend turns down
            if i > 0 and not np.isnan(trix[i-1]) and trix[i] < trix[i-1] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns up or trend turns up
            if i > 0 and not np.isnan(trix[i-1]) and trix[i] > trix[i-1] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals