# 6h_Keltner_Bollinger_Squeeze_Breakout_1dTrend_VolumeSpike
# Hypothesis: Keltner/Bollinger squeeze identifies low volatility breakout points on 6h timeframe.
# Breakout confirmed by price closing outside Keltner bands with volume spike and 1d trend alignment.
# Works in bull markets (breakouts continue) and bear markets (mean reversion squeezes before breakdowns).
# Targets 15-30 trades/year per symbol with disciplined entries.

name = "6h_Keltner_Bollinger_Squeeze_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and Bollinger/Keltner components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate Bollinger Bands (20, 2) on 1d
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d

    # Calculate Keltner Channels (20, 1.5) on 1d using ATR
    atr_period = 20
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first period
    atr_1d = pd.Series(tr_1d).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    ema_mid_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_kc_1d = ema_mid_1d + 1.5 * atr_1d
    lower_kc_1d = ema_mid_1d - 1.5 * atr_1d

    # Squeeze condition: Bollinger Bands inside Keltner Channels
    squeeze_condition = (upper_bb_1d <= upper_kc_1d) & (lower_bb_1d >= lower_kc_1d)

    # Align 1d indicators to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    upper_kc_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_kc_1d)
    lower_kc_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_kc_1d)

    # Volume spike: volume > 2.0 * 20-period average (~80 periods = 20 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(upper_bb_1d_aligned[i]) or
            np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(upper_kc_1d_aligned[i]) or
            np.isnan(lower_kc_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Squeeze breakout above upper KC + uptrend + volume spike
            if close[i] > upper_kc_1d_aligned[i] and squeeze_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Squeeze breakout below lower KC + downtrend + volume spike
            elif close[i] < lower_kc_1d_aligned[i] and squeeze_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Keltner Channel or trend turns bearish
            if close[i] < lower_kc_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Keltner Channel or trend turns bullish
            if close[i] > upper_kc_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals