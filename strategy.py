#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h ADX trend filter and volume confirmation
# Uses Williams Fractals (1d) for precise swing points, breaks above/below recent fractals with 12h ADX > 25 for trend strength
# Volume spike confirms breakout momentum. Designed to catch strong trends in both bull and bear markets
# Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals (5-point: bar high/low > 2 bars on each side)
    def calculate_williams_fractals(high_arr, low_arr):
        n1 = len(high_arr)
        bullish = np.zeros(n1, dtype=bool)
        bearish = np.zeros(n1, dtype=bool)
        
        for i in range(2, n1 - 2):
            # Bullish fractal: low[i] is lowest among i-2, i-1, i, i+1, i+2
            if (low[i] < low[i-2] and low[i] < low[i-1] and 
                low[i] < low[i+1] and low[i] < low[i+2]):
                bullish[i] = True
            # Bearish fractal: high[i] is highest among i-2, i-1, i, i+1, i+2
            if (high[i] > high[i-2] and high[i] > high[i-1] and 
                high[i] > high[i+1] and high[i] > high[i+2]):
                bearish[i] = True
        return bullish, bearish
    
    bullish_fractal, bearish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Convert to price levels (use the fractal point value)
    bullish_fractal_val = np.where(bullish_fractal, low_1d, np.nan)
    bearish_fractal_val = np.where(bearish_fractal, high_1d, np.nan)
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        n1 = len(high_arr)
        if n1 < period + 1:
            return np.full(n1, np.nan)
        
        # True Range
        tr1 = np.abs(high_arr - low_arr)
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First value has no previous close
        
        # Directional Movement
        up_move = high_arr - np.roll(high_arr, 1)
        down_move = np.roll(low_arr, 1) - low_arr
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
        def wilders_smooth(arr, period):
            n1 = len(arr)
            result = np.full(n1, np.nan)
            if n1 < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(arr[:period]) / period
            # Subsequent values: Wilder smoothing
            for i in range(period, n1):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_sum = wilders_smooth(tr, period)
        plus_dm_sum = wilders_smooth(plus_dm, period)
        minus_dm_sum = wilders_smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_sum != 0, 100 * plus_dm_sum / tr_sum, 0)
        minus_di = np.where(tr_sum != 0, 100 * minus_dm_sum / tr_sum, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 6-hour timeframe
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_val, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_val, additional_delay_bars=2)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above recent bearish fractal + ADX > 25 + volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                adx_12h_aligned[i] > 25 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below recent bullish fractal + ADX > 25 + volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  adx_12h_aligned[i] > 25 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite fractal level or ADX weakens
            if position == 1:
                if (close[i] < bullish_fractal_aligned[i] or 
                    adx_12h_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > bearish_fractal_aligned[i] or 
                    adx_12h_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_12hADX_Volume_Session"
timeframe = "6h"
leverage = 1.0