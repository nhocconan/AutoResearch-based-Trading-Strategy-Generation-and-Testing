#!/usr/bin/env python3
# 160114: 1h_Momentum_Reversal_4hTrend_1dVolatility
# Hypothesis: In ranging or mean-reverting conditions (1d volatility low), use 1h momentum reversals (price crossing 20-period EMA with RSI divergence) aligned to 4h trend direction. This captures short-term mean reversion within the larger trend, reducing false signals in chop while maintaining trend alignment. Works in bull/bear by following 4h EMA50 trend and avoiding trades when 1d volatility is high (trending markets prone to false reversals).

name = "1h_Momentum_Reversal_4hTrend_1dVolatility"
timeframe = "1h"
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

    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values

    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # 1h EMA20 for mean reversion signal
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # 1h RSI(14) for momentum exhaustion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # 1d ATR(14) normalized by price for volatility regime
    atr_14_1h = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    atr_ma_14 = pd.Series(atr_14_1h).rolling(window=14, min_periods=14).mean().values
    volatility_ratio = atr_14_1h / atr_ma_14
    volatility_ratio = np.where(atr_ma_14 > 0, volatility_ratio, 1.0)

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(rsi[i]) or np.isnan(volatility_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: only trade when volatility is low (mean-reversion regime)
        vol_filter = volatility_ratio[i] < 0.8

        if position == 0:
            # LONG: Price crosses above EMA20 + RSI < 30 (oversold) + 4h uptrend + low vol
            if (close[i] > ema_20[i] and close[i-1] <= ema_20[i-1] and
                rsi[i] < 30 and ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and
                vol_filter):
                signals[i] = 0.20
                position = 1
            # SHORT: Price crosses below EMA20 + RSI > 70 (overbought) + 4h downtrend + low vol
            elif (close[i] < ema_20[i] and close[i-1] >= ema_20[i-1] and
                  rsi[i] > 70 and ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and
                  vol_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA20 OR RSI > 70
            if (close[i] < ema_20[i] and close[i-1] >= ema_20[i-1]) or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA20 OR RSI < 30
            if (close[i] > ema_20[i] and close[i-1] <= ema_20[i-1]) or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals