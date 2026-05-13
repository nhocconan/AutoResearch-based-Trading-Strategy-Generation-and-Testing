#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray + ADX regime filter with 1d trend confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and Bear Power < 0 (both bullish) AND ADX > 25 (trending) AND close > 1d EMA50.
# Short when Bear Power > 0 and Bull Power < 0 (both bearish) AND ADX > 25 AND close < 1d EMA50.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Elder Ray measures bull/bear strength relative to EMA13; ADX filters for trending markets only.
# 1d EMA50 ensures higher timeframe trend alignment, reducing counter-trend whipsaws.
# This combination should work in both bull and bear markets by only taking strong trend-following signals.

name = "6h_ElderRay_ADX_Regime_1dEMA50_Trend"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate ADX (14-period)
    lookback_adx = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initial values
    atr[lookback_adx] = np.mean(tr[1:lookback_adx+1])
    plus_di[lookback_adx] = (np.sum(plus_dm[1:lookback_adx+1]) / atr[lookback_adx]) * 100
    minus_di[lookback_adx] = (np.sum(minus_dm[1:lookback_adx+1]) / atr[lookback_adx]) * 100
    
    for i in range(lookback_adx + 1, n):
        atr[i] = (atr[i-1] * (lookback_adx - 1) + tr[i]) / lookback_adx
        plus_di[i] = (plus_di[i-1] * (lookback_adx - 1) + plus_dm[i]) / atr[i] * 100
        minus_di[i] = (minus_di[i-1] * (lookback_adx - 1) + minus_dm[i]) / atr[i] * 100
        dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
    
    # ADX is smoothed DX
    adx[lookback_adx*2] = np.mean(dx[lookback_adx+1:lookback_adx*2+1])
    for i in range(lookback_adx*2 + 1, n):
        adx[i] = (adx[i-1] * (lookback_adx - 1) + dx[i]) / lookback_adx
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback_adx * 2  # Need enough data for ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (both bullish) AND ADX > 25 AND close > 1d EMA50
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                adx[i] > 25 and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 AND Bull Power < 0 (both bearish) AND ADX > 25 AND close < 1d EMA50
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  adx[i] > 25 and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Either Elder Ray turns bearish OR ADX < 20 (trend weak) OR close crosses below 1d EMA50
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                adx[i] < 20 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Either Elder Ray turns bullish OR ADX < 20 OR close crosses above 1d EMA50
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                adx[i] < 20 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray + ADX regime filter with 1d trend confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and Bear Power < 0 (both bullish) AND ADX > 25 (trending) AND close > 1d EMA50.
# Short when Bear Power > 0 and Bull Power < 0 (both bearish) AND ADX > 25 AND close < 1d EMA50.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Elder Ray measures bull/bear strength relative to EMA13; ADX filters for trending markets only.
# 1d EMA50 ensures higher timeframe trend alignment, reducing counter-trend whipsaws.
# This combination should work in both bull and bear markets by only taking strong trend-following signals.

name = "6h_ElderRay_ADX_Regime_1dEMA50_Trend"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate ADX (14-period)
    lookback_adx = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initial values
    atr[lookback_adx] = np.mean(tr[1:lookback_adx+1])
    plus_di[lookback_adx] = (np.sum(plus_dm[1:lookback_adx+1]) / atr[lookback_adx]) * 100
    minus_di[lookback_adx] = (np.sum(minus_dm[1:lookback_adx+1]) / atr[lookback_adx]) * 100
    
    for i in range(lookback_adx + 1, n):
        atr[i] = (atr[i-1] * (lookback_adx - 1) + tr[i]) / lookback_adx
        plus_di[i] = (plus_di[i-1] * (lookback_adx - 1) + plus_dm[i]) / atr[i] * 100
        minus_di[i] = (minus_di[i-1] * (lookback_adx - 1) + minus_dm[i]) / atr[i] * 100
        dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
    
    # ADX is smoothed DX
    adx[lookback_adx*2] = np.mean(dx[lookback_adx+1:lookback_adx*2+1])
    for i in range(lookback_adx*2 + 1, n):
        adx[i] = (adx[i-1] * (lookback_adx - 1) + dx[i]) / lookback_adx
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback_adx * 2  # Need enough data for ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (both bullish) AND ADX > 25 AND close > 1d EMA50
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                adx[i] > 25 and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 AND Bull Power < 0 (both bearish) AND ADX > 25 AND close < 1d EMA50
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  adx[i] > 25 and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Either Elder Ray turns bearish OR ADX < 20 (trend weak) OR close crosses below 1d EMA50
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                adx[i] < 20 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Either Elder Ray turns bullish OR ADX < 20 OR close crosses above 1d EMA50
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                adx[i] < 20 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals