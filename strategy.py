#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 12h ADX trend filter + volume confirmation
# - Long when price breaks above Camarilla R4 AND 12h ADX > 25 AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla S4 AND 12h ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price crosses Camarilla PP (pivot point) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels
# - 12h ADX ensures we trade only when higher timeframe has strong trend
# - Volume confirmation reduces false breakouts

name = "6h_12h_camarilla_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Pre-compute 6h Camarilla levels from previous day
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points from previous day's OHLC
    # We need to resample to 1d to get daily OHLC, then calculate Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get daily OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r4 = np.full_like(d_high, np.nan)
    camarilla_s4 = np.full_like(d_high, np.nan)
    camarilla_pp = np.full_like(d_high, np.nan)
    
    for i in range(len(d_high)):
        if not (np.isnan(d_high[i]) or np.isnan(d_low[i]) or np.isnan(d_close[i])):
            camarilla_pp[i] = (d_high[i] + d_low[i] + d_close[i]) / 3
            camarilla_r4[i] = camarilla_pp[i] + 1.1 * (d_high[i] - d_low[i])
            camarilla_s4[i] = camarilla_pp[i] - 1.1 * (d_high[i] - d_low[i])
    
    # Align daily Camarilla levels to 6h timeframe (using previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4, additional_delay_bars=1)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4, additional_delay_bars=1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp, additional_delay_bars=1)
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period]) / period
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla R4 AND 12h ADX > 25 AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla S4 AND 12h ADX > 25 AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla PP OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < camarilla_pp_aligned[i] or close[i] < camarilla_s4_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > camarilla_pp_aligned[i] or close[i] > camarilla_r4_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals