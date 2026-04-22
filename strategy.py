#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout (20) + 12h ADX trend filter + volume confirmation
    # Donchian channels provide clear breakout signals with defined stop levels
    # ADX > 25 filters for trending markets only, avoiding whipsaws in ranges
    # Volume confirmation ensures breakouts have institutional participation
    # Works in both bull/bear markets by trading breakouts in direction of 12h trend
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (similar to EMA with alpha=1/period)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.mean(data[:period])
                # Subsequent values: prev*(period-1)/period + current/period
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = WilderSmooth(tr, period)
        plus_di = 100 * WilderSmooth(plus_dm, period) / atr
        minus_di = 100 * WilderSmooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Donchian channels (20-period) on 6h data
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20  # 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian upper + ADX > 25 (trending) + volume surge
            if close[i] > donchian_upper[i] and adx_12h_aligned[i] > 25 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian lower + ADX > 25 (trending) + volume surge
            elif close[i] < donchian_lower[i] and adx_12h_aligned[i] > 25 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back through Donchian middle or ADX drops below 20
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if position == 1:
                if close[i] < donchian_middle or adx_12h_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_middle or adx_12h_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_12hADX_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0