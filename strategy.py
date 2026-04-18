#!/usr/bin/env python3
"""
4h_Choppiness_Keltner_Reversion
Hypothesis: In sideways markets (Choppiness > 61.8), price tends to revert to the Keltner middle (EMA20). 
Long when price touches lower band with bullish engulfing candle, short when touches upper band with bearish engulfing.
Works in both bull/bear markets because range-bound periods occur in all regimes. Uses 1d ADX to avoid strong trends.
"""

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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA for Keltner middle
    ema_20 = np.zeros(n)
    ema_20[:] = np.nan
    if n >= 20:
        k = 2 / (20 + 1)
        ema_20[19] = np.mean(close[:20])
        for i in range(20, n):
            ema_20[i] = close[i] * k + ema_20[i-1] * (1 - k)
    
    # Calculate ATR(10) for Keltner width
    atr = np.zeros(n)
    atr[:] = np.nan
    if n >= 2:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder's smoothing for ATR
        atr[9] = np.mean(tr[:10])
        for i in range(10, n):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Keltner bands (multiplier = 1.5)
    keltner_upper = ema_20 + 1.5 * atr
    keltner_lower = ema_20 - 1.5 * atr
    
    # Choppiness Index (14-period) - range detection
    def calculate_choppiness(high_arr, low_arr, close_arr, period=14):
        chop = np.full(len(close_arr), np.nan)
        if len(close_arr) < period * 2:
            return chop
        atr_sum = np.zeros(len(close_arr))
        tr = np.zeros(len(close_arr))
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(tr)):
            tr[i] = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i-1]), abs(low_arr[i] - close_arr[i-1]))
        # Wilder's smoothing for TR sum
        atr_sum[period-1] = np.sum(tr[:period])
        for i in range(period, len(tr)):
            atr_sum[i] = atr_sum[i-1] - (atr_sum[i-1] / period) + tr[i]
        # Highest high and lowest low over period
        hh = np.zeros(len(close_arr))
        ll = np.zeros(len(close_arr))
        for i in range(len(close_arr)):
            if i < period:
                hh[i] = np.max(high_arr[:i+1])
                ll[i] = np.min(low_arr[:i+1])
            else:
                hh[i] = np.max(high_arr[i-period+1:i+1])
                ll[i] = np.min(low_arr[i-period+1:i+1])
        # Choppiness formula
        for i in range(period-1, len(close_arr)):
            if atr_sum[i] > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bull_engulf = np.zeros(n, dtype=bool)
    bear_engulf = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if close[i] > open_[i] and close[i-1] < open_[i-1]:  # current up, previous down
            if close[i] >= open_[i-1] and open_[i] <= close[i-1]:
                bull_engulf[i] = True
        if close[i] < open_[i] and close[i-1] > open_[i-1]:  # current down, previous up
            if close[i] <= open_[i-1] and open_[i] >= close[i-1]:
                bear_engulf[i] = True
    
    # Need open prices
    open_ = prices['open'].values
    
    # Recalculate engulfing with correct open
    bull_engulf = np.zeros(n, dtype=bool)
    bear_engulf = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if close[i] > open_[i] and close[i-1] < open_[i-1]:  # current up, previous down
            if close[i] >= open_[i-1] and open_[i] <= close[i-1]:
                bull_engulf[i] = True
        if close[i] < open_[i] and close[i-1] > open_[i-1]:  # current down, previous up
            if close[i] <= open_[i-1] and open_[i] >= close[i-1]:
                bear_engulf[i] = True
    
    # 1d ADX for trend strength filter (avoid strong trends)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        adx = np.full(len(close_arr), np.nan)
        if len(close_arr) < period * 2:
            return adx
        # True Range
        tr = np.zeros(len(close_arr))
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(tr)):
            tr[i] = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i-1]), abs(low_arr[i] - close_arr[i-1]))
        # Directional Movement
        plus_dm = np.zeros(len(close_arr))
        minus_dm = np.zeros(len(close_arr))
        for i in range(1, len(high_arr)):
            up_move = high_arr[i] - high_arr[i-1]
            down_move = low_arr[i-1] - low_arr[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        # Smoothed TR, PlusDM, MinusDM (Wilder's smoothing)
        atr_adx = np.zeros(len(close_arr))
        plus_dm_smooth = np.zeros(len(close_arr))
        minus_dm_smooth = np.zeros(len(close_arr))
        atr_adx[period-1] = np.sum(tr[:period])
        plus_dm_smooth[period-1] = np.sum(plus_dm[:period])
        minus_dm_smooth[period-1] = np.sum(minus_dm[:period])
        for i in range(period, len(tr)):
            atr_adx[i] = (atr_adx[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        # Directional Indicators
        plus_di = np.zeros(len(close_arr))
        minus_di = np.zeros(len(close_arr))
        dx = np.zeros(len(close_arr))
        for i in range(period-1, len(close_arr)):
            if atr_adx[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr_adx[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr_adx[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        # ADX = smoothed DX
        adx[2*period-2] = np.mean(dx[period-1:2*period-1]) if 2*period-1 <= len(dx) else np.nan
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(chop[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range market: Choppiness > 61.8, weak trend: ADX < 25
        if chop[i] > 61.8 and adx_1d_aligned[i] < 25:
            # Long: price at lower Keltner band with bullish engulfing
            if low[i] <= keltner_lower[i] and bull_engulf[i]:
                if position <= 0:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25  # maintain
            # Short: price at upper Keltner band with bearish engulfing
            elif high[i] >= keltner_upper[i] and bear_engulf[i]:
                if position >= 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25  # maintain
            # Exit: price returns to middle (EMA20) or conditions change
            elif position == 1 and (close[i] >= ema_20[i] or chop[i] < 50 or adx_1d_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (close[i] <= ema_20[i] or chop[i] < 50 or adx_1d_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # Trending or choppy but not extreme range: stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "4h_Choppiness_Keltner_Reversion"
timeframe = "4h"
leverage = 1.0