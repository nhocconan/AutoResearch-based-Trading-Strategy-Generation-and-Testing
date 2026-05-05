#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 12h ADX trend filter + volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when: Bull Power > 0 AND Bear Power < 0 (bullish imbalance) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Short when: Bear Power > 0 AND Bull Power < 0 (bearish imbalance) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Exit when: Elder Ray imbalance reverses OR ADX < 20 (trend weakens)
# Uses Elder Ray for momentum imbalance, ADX for regime filter, volume for conviction
# Timeframe: 6h, HTF: 12h for ADX. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_12hADX_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(13) for Elder Ray on 6h
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # Elder Ray imbalance signals
    bullish_imbalance = (bull_power > 0) & (bear_power < 0)  # True bullish momentum
    bearish_imbalance = (bear_power > 0) & (bull_power < 0)  # True bearish momentum
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    if len(high_12h) >= 14:
        # True Range
        tr1 = np.abs(high_12h[1:] - low_12h[1:])
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
        minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, 14)
    else:
        adx = np.full(len(df_12h), np.nan)
    
    # ADX trend filter: ADX > 25 = strong trend
    adx_trend = adx > 25
    adx_weak = adx < 20  # for exit condition
    
    # Align 12h ADX to 6h timeframe
    adx_trend_aligned = align_htf_to_ltf(prices, df_12h, adx_trend.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_12h, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bullish_imbalance[i]) or np.isnan(bearish_imbalance[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_trend_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish imbalance + strong trend + volume filter
            if (bullish_imbalance[i] and 
                adx_trend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish imbalance + strong trend + volume filter
            elif (bearish_imbalance[i] and 
                  adx_trend_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: imbalance reverses OR trend weakens
            if (not bullish_imbalance[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: imbalance reverses OR trend weakens
            if (not bearish_imbalance[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals