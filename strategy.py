#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and ATR-based volatility filter.
# Long when price breaks above Donchian upper channel AND 12h EMA50 > EMA200 (bullish trend) AND ATR(14) < ATR(50) (low volatility regime).
# Short when price breaks below Donchian lower channel AND 12h EMA50 < EMA200 (bearish trend) AND ATR(14) < ATR(50) (low volatility regime).
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe with strict entry conditions.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h.

name = "4h_Donchian20_Breakout_12hEMA_Trend_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 and EMA200 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_trend = align_htf_to_ltf(prices, df_12h, ema_50 > ema_200)  # Boolean: True for bullish, False for bearish
    
    # Calculate ATR(14) and ATR(50) for volatility filter (LTF)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # < 1.0 indicates low volatility regime
    vol_filter = atr_ratio < 1.0  # Low volatility condition
    
    # Calculate Donchian(20) channels (LTF)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2  # Midpoint for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_trend[i]) or 
            np.isnan(vol_filter[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND 12h EMA50 > EMA200 AND low volatility
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                ema_trend[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND 12h EMA50 < EMA200 AND low volatility
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  not ema_trend[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals