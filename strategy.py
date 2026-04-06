#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX + Directional Movement for trend strength + Williams %R for overbought/oversold entries
# Long when ADX > 25 (strong trend) AND +DI > -DI (bullish) AND Williams %R < -80 (oversold pullback)
# Short when ADX > 25 AND -DI > +DI (bearish) AND Williams %R > -20 (overbought pullback)
# Exit when ADX < 20 (weakening trend) or Williams %R crosses back through -50
# Uses weekly trend filter: only take longs when price > weekly EMA(50), shorts when price < weekly EMA(50)
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_adx_williamsr_weekly_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX + DI calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period Wilder's smoothing = EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 14)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high.values - close) / (highest_high.values - lowest_low.values)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high.values - lowest_low.values) == 0, -50, williams_r)
    
    # Weekly trend filter: EMA(50) on weekly closes
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(williams_r[i]) or np.isnan(weekly_ema_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: ADX weakening or Williams %R crosses -50
        if position == 1:  # long position
            if adx[i] < 20 or williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if adx[i] < 20 or williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter
            # Long: ADX > 25 (strong trend) AND +DI > -DI (bullish) AND Williams %R < -80 (oversold) AND price > weekly EMA
            if (adx[i] > 25 and plus_di[i] > minus_di[i] and williams_r[i] < -80 and 
                close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 AND -DI > +DI (bearish) AND Williams %R > -20 (overbought) AND price < weekly EMA
            elif (adx[i] > 25 and minus_di[i] > plus_di[i] and williams_r[i] > -20 and 
                  close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals