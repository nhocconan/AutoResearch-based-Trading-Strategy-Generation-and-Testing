#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 level AND price > EMA20(4h) AND volume > 1.8x 20-period average
# Short when price breaks below Camarilla S1 level AND price < EMA20(4h) AND volume > 1.8x 20-period average
# Exit when price returns to Camarilla pivot level (mean reversion)
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-37 trades/year for 1h timeframe.
# 4h EMA20 provides higher timeframe trend filter to avoid counter-trend whipsaws.
# Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values  # datetime64[ms]
    
    # Pre-compute session filter (08-20 UTC) - avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla pivot levels for each 1h bar using previous 4h OHLC
    # We need the previous completed 4h bar's OHLC for each 1h bar
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    if len(df_4h) >= 1:
        # Get previous completed 4h bar's OHLC (shift by 1 to avoid look-ahead)
        prev_close_4h = np.roll(close_4h, 1)
        prev_open_4h = np.roll(df_4h['open'].values, 1)
        prev_high_4h = np.roll(df_4h['high'].values, 1)
        prev_low_4h = np.roll(df_4h['low'].values, 1)
        # Set first value to NaN since there's no previous bar
        prev_close_4h[0] = np.nan
        prev_open_4h[0] = np.nan
        prev_high_4h[0] = np.nan
        prev_low_4h[0] = np.nan
        
        # Calculate Camarilla levels from previous 4h bar
        camarilla_pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
        camarilla_range = prev_high_4h - prev_low_4h
        camarilla_r1_4h = camarilla_pivot_4h + camarilla_range * 1.1 / 12
        camarilla_s1_4h = camarilla_pivot_4h - camarilla_range * 1.1 / 12
        
        # Align to 1h timeframe (these levels are valid after the 4h bar completes)
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
        camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
        camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
        
        camarilla_pivot = camarilla_pivot_aligned
        camarilla_r1 = camarilla_r1_aligned
        camarilla_s1 = camarilla_s1_aligned
    
    # Volume confirmation: volume > 1.8x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(camarilla_pivot[i]) or 
            np.isnan(volume_filter[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND price > EMA20(4h) AND volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_20_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND price < EMA20(4h) AND volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_20_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot (mean reversion)
            if close[i] <= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla pivot (mean reversion)
            if close[i] >= camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals