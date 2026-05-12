#!/usr/bin/env python3
# 6h_Volatility_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Bollinger Band squeeze (low volatility) on 6m followed by breakout with volume
# and alignment to daily trend (EMA50) captures explosive moves in both bull and bear markets.
# Works by identifying periods of low volatility (BB width < 20th percentile) and entering
# on breakouts above/below the bands with volume confirmation and trend filter.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

name = "6h_Volatility_Squeeze_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Bollinger Bands on 6m (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = (upper - lower) / sma  # normalized width

    # BB width percentile (20-period lookback) to detect squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze = bb_width_percentile < 20  # in lowest 20% = squeeze

    # Daily EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: current volume > 1.8x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(squeeze[i])):
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
            # Only trade on breakout after squeeze
            if squeeze[i-1] and not squeeze[i]:  # squeeze just released
                # LONG: Break above upper band with volume and uptrend
                if close[i] > upper[i] and price_above_daily_ema and volume_ok[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Break below lower band with volume and downtrend
                elif close[i] < lower[i] and price_below_daily_ema and volume_ok[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below middle Bollinger Band (mean reversion) or trend turns down
            if close[i] < sma[i] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above middle Bollinger Band or trend turns up
            if close[i] > sma[i] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals