#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d EMA trend filter and volume confirmation.
# Camarilla levels provide strong intraday support/resistance; breakouts indicate momentum.
# 1d EMA34 filters for higher timeframe trend direction to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction. Designed for both bull and bear markets
# by following the higher timeframe trend.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla levels from previous day (using 1d data) ===
    # Calculate from previous day's OHLC (shifted by 1 to avoid look-ahead)
    # We'll calculate the levels inside the loop using previous day's data
    # But first, get the 1d data arrays
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values  # already have
    
    # === 4h ATR for stoploss (optional, using signal for exit) ===
    # We'll use a simple time-based exit or reversal signal for now
    # For risk management, we rely on the signal flipping to 0 on reversal
    
    # Volume confirmation (20-period average on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA and volume MA are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for the CURRENT day using previous day's OHLC
        # We need the previous day's data: index in 1d array for the day before the current 4h bar's day
        current_time = prices['open_time'].iloc[i]
        # Find the start of the current day
        current_day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        # Previous day's start
        prev_day_start = current_day_start - pd.Timedelta(days=1)
        
        # Find index of previous day's OHLC in 1d arrays
        # Since 1d bars are at 00:00 UTC, we can find the index where open_time == prev_day_start
        # We'll use searchsorted on the 1d open_time array
        # But to avoid complexity and look-ahead, we'll use the last completed 1d bar
        # At any 4h bar, the last completed 1d bar is the one for the previous day if current time < 00:00
        # Actually, simpler: use the 1d bar that completed at 00:00 UTC of the current day
        # which is the same as the previous day's data from perspective of current 4h bar
        # We'll use index = number of completed 1d bars so far
        # We can track this by counting days, but easier: use the 1d index corresponding to
        # the date of the current 4h bar minus 1 day
        
        # Instead, let's pre-calculate the Camarilla levels for each 1d bar and align them
        # This is cleaner and avoids look-ahead
        # Calculate typical Camarilla levels for each 1d bar based on that day's OHLC
        # Then shift by 1 to use previous day's levels (to avoid look-ahead)
        # Then align to 4h
        
        # We'll do this outside the loop for efficiency
        pass  # We'll implement below
    
    # Instead of calculating in loop, let's precompute
    # Recompute outside loop for clarity and performance
    
    # Recalculate: we need to compute Camarilla for each 1d bar, then use previous day's
    # Actually, Camarilla levels for a given day are based on that day's OHLC, used for next day
    # So for trading on day T, we use Camarilla levels from day T-1
    
    # Let's recompute everything outside the loop
    
    # Exit and rewrite the function properly
    
    # Re-start: compute everything we can outside the loop
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Previous day's Camarilla levels ===
    # For each 1d bar, calculate Camarilla levels based on its OHLC
    # These levels are valid for the next day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Typical Camarilla multiplier
    # R3/S3 = C ± (H-L) * 1.1/2
    # We'll calculate R3 and S3
    camarilla_r3_1d = close_1d_arr + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3_1d = close_1d_arr - (high_1d - low_1d) * 1.1 / 2
    
    # Shift by 1 to get previous day's levels (to avoid look-ahead)
    # For the first day, we have no previous day, so we'll have NaN
    camarilla_r3_prev = np.roll(camarilla_r3_1d, 1)
    camarilla_s3_prev = np.roll(camarilla_s3_1d, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    
    # Align previous day's Camarilla levels to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_prev)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_prev)
    
    # === Volume confirmation (20-period average on 4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA and volume MA are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_above_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: breakout above R3, above EMA (uptrend), volume confirmation
            if breakout_above_r3 and price_above_ema and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below S3, below EMA (downtrend), volume confirmation
            elif breakout_below_s3 and price_below_ema and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: break below S3 or trend turns down
            if breakout_below_s3 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above R3 or trend turns up
            if breakout_above_r3 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals