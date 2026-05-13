#!/usr/bin/env python3
# 4h_ADX_RSI_Pullback_1dTrend_Volume
# Hypothesis: In strong trends (ADX>25), buy pullbacks to RSI(14)<40 in uptrends or sell rallies to RSI>60 in downtrends, 
# confirmed by 1d EMA50 trend and volume spike. Works in bull/bear by following the 1d trend.
# Target: 20-40 trades/year on 4h to minimize fee drag.

name = "4h_ADX_RSI_Pullback_1dTrend_Volume"
timeframe = "4h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(low, 1)), np.abs(low - np.roll(high, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values

    # RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: ADX>25 (strong trend) + RSI<40 (pullback) + price>EMA50_1d (uptrend) + volume spike
            if (adx[i] > 25 and 
                rsi[i] < 40 and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: ADX>25 + RSI>60 (pullback in downtrend) + price<EMA50_1d (downtrend) + volume spike
            elif (adx[i] > 25 and 
                  rsi[i] > 60 and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI>60 (overbought) or trend weakens (ADX<20)
            if rsi[i] > 60 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI<40 (oversold) or trend weakens (ADX<20)
            if rsi[i] < 40 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
#!/usr/bin/env python3
# 4h_ADX_RSI_Pullback_1dTrend_Volume
# Hypothesis: In strong trends (ADX>25), buy pullbacks to RSI(14)<40 in uptrends or sell rallies to RSI>60 in downtrends, 
# confirmed by 1d EMA50 trend and volume spike. Works in bull/bear by following the 1d trend.
# Target: 20-40 trades/year on 4h to minimize fee drag.

name = "4h_ADX_RSI_Pullback_1dTrend_Volume"
timeframe = "4h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(low, 1)), np.abs(low - np.roll(high, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values

    # RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: ADX>25 (strong trend) + RSI<40 (pullback) + price>EMA50_1d (uptrend) + volume spike
            if (adx[i] > 25 and 
                rsi[i] < 40 and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: ADX>25 + RSI>60 (pullback in downtrend) + price<EMA50_1d (downtrend) + volume spike
            elif (adx[i] > 25 and 
                  rsi[i] > 60 and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI>60 (overbought) or trend weakens (ADX<20)
            if rsi[i] > 60 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI<40 (oversold) or trend weakens (ADX<20)
            if rsi[i] < 40 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals