#!/usr/bin/env python3
"""
6h_ADX_Trend_Strength_with_Volume_Confirmation
Hypothesis: Strong trends identified by ADX(14) > 25 combined with volume confirmation from daily timeframe provide reliable directional bias. 
Entry when price is above/below EMA(50) and volume is above 1.5x its 20-period average on daily timeframe. 
Exit when ADX weakens (< 20) or price crosses EMA(50) in opposite direction.
Designed for 15-35 trades/year on 6h timeframe to minimize fee decay while capturing sustained moves in bull/bear markets.
"""

name = "6h_ADX_Trend_Strength_with_Volume_Confirmation"
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
    volume = prices['volume'].values

    # Get daily data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate daily volume average
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate EMA(50) on 6h close
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values

    # Calculate ADX(14) on 6h data
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        adx_val = adx[i]
        ema_val = ema_50[i]
        vol_avg_val = vol_avg_20_1d_aligned[i]
        vol_1d_today = volume_1d[i // 24] if i // 24 < len(volume_1d) else volume_1d[-1]

        if np.isnan(adx_val) or np.isnan(ema_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Strong uptrend (ADX > 25) + price above EMA + volume surge
            if adx_val > 25 and plus_di[i] > minus_di[i] and close[i] > ema_val and vol_1d_today > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Strong downtrend (ADX > 25) + price below EMA + volume surge
            elif adx_val > 25 and minus_di[i] > plus_di[i] and close[i] < ema_val and vol_1d_today > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakens or price crosses below EMA
            if adx_val < 20 or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens or price crosses above EMA
            if adx_val < 20 or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals