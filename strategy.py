#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour ADX trend filter with 4-hour EMA trend and volume confirmation
# Long when: ADX(14) > 25, 1h EMA(20) > 1h EMA(50), 4h EMA(34) > 4h EMA(34) prev, volume > 1.5x 20-period avg
# Short when: ADX(14) > 25, 1h EMA(20) < 1h EMA(50), 4h EMA(34) < 4h EMA(34) prev, volume > 1.5x 20-period avg
# Uses 4h EMA trend as primary filter to avoid counter-trend trades, ADX to ensure trending conditions
# Volume confirms momentum. Target: 15-30 trades/year to stay within fee limits.
# ADX helps avoid choppy markets where trend following fails.

name = "1h_ADX_4hEMA_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_prev = np.roll(ema34_4h, 1)
    ema34_4h_prev[0] = np.nan
    ema34_4h_up = ema34_4h > ema34_4h_prev
    ema34_4h_up_prev = np.roll(ema34_4h_up, 1)
    ema34_4h_up_prev[0] = False
    ema34_4h_up_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h_up)
    
    # Calculate 1h EMA(20) and EMA(50) for trend
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_above_ema50 = ema20 > ema50
    
    # Calculate 1h ADX(14) for trend strength
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    
    # Pad arrays to original length
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    tr = np.concatenate([[np.nan], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    smoothed_plus_dm = wilder_smoothing(plus_dm, 14)
    smoothed_minus_dm = wilder_smoothing(minus_dm, 14)
    smoothed_tr = wilder_smoothing(tr, 14)
    
    # Avoid division by zero
    plus_di = np.where(smoothed_tr != 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    minus_di = np.where(smoothed_tr != 0, 100 * smoothed_minus_dm / smoothed_tr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smoothing(dx, 14)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema34_4h_up_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        ema20_val = ema20[i]
        ema50_val = ema50[i]
        ema34_4h_up_val = ema34_4h_up_aligned[i]
        vol_conf = volume_confirmed[i]
        session_ok = session_filter[i]
        
        if position == 0:
            # Enter long: ADX > 25, EMA20 > EMA50, 4h EMA up, volume confirmed, in session
            if (adx_val > 25 and ema20_val > ema50_val and ema34_4h_up_val and 
                vol_conf and session_ok):
                signals[i] = 0.20
                position = 1
            # Enter short: ADX > 25, EMA20 < EMA50, 4h EMA down, volume confirmed, in session
            elif (adx_val > 25 and ema20_val < ema50_val and not ema34_4h_up_val and 
                  vol_conf and session_ok):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 OR EMA20 < EMA50 OR 4h EMA turns down
            if (adx_val < 20 or ema20_val < ema50_val or not ema34_4h_up_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: ADX < 20 OR EMA20 > EMA50 OR 4h EMA turns up
            if (adx_val < 20 or ema20_val > ema50_val or ema34_4h_up_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals