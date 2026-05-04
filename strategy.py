#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength relative to EMA13.
# 12h ADX > 25 filters for trending markets to avoid whipsaw in ranges.
# Long when Bull Power > 0 and ADX > 25; Short when Bear Power > 0 and ADX > 25.
# Exit when power reverses or ADX < 20 (range regime).
# Works in bull markets by capturing strong uptrends via Bull Power; in bear markets via Bear Power shorts.
# The 12h ADX regime filter reduces false signals during consolidation, improving win rate.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_ElderRay_12hADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        data = np.asarray(data)
        length = len(data)
        result = np.full(length, np.nan)
        if length < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(data[:period])
        # subsequent values: Wilder's smoothing
        for i in range(period, length):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr_14 = wilders_smooth(tr, 14)
    plus_dm_14 = wilders_smooth(plus_dm, 14)
    minus_dm_14 = wilders_smooth(minus_dm, 14)
    
    # +DI and -DI
    plus_di = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    # Align 12h ADX to 6h timeframe (wait for completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: 12h ADX > 25 = trending market
        is_trending = adx_aligned[i] > 25
        # Exit regime: ADX < 20 = range (hysteresis to avoid whipsaw)
        is_range = adx_aligned[i] < 20
        
        if position == 0:
            if is_trending:
                # Long: Bull Power > 0 (bulls in control)
                if bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 (bears in control)
                elif bear_power[i] > 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bear Power > 0 (bears take over) OR range regime
            if bear_power[i] > 0 or is_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (bulls take over) OR range regime
            if bull_power[i] > 0 or is_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals