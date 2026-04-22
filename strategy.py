#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation
    # Donchian breakouts capture momentum in trending markets
    # 1w ADX > 25 filters for strong trends, avoiding sideways chop
    # Volume confirmation ensures breakouts have institutional participation
    # This combination works in both bull (breakouts up) and bear (breakouts down) markets
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for ADX trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on weekly data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.mean(data[:period])
                # Subsequent values: prev*(1-alpha) + current*alpha
                for i in range(period, len(data)):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx[:] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[np.isnan(plus_di) | np.isnan(minus_di) | (plus_di + minus_di) == 0] = np.nan
        
        # ADX is smoothed DX
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Donchian Channel (20-period) on 12h data
    def donchian_channel(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(len(high)):
            if i >= period - 1:
                upper[i] = np.max(high[i - period + 1:i + 1])
                lower[i] = np.min(low[i - period + 1:i + 1])
        return upper, lower
    
    donch_upper, donch_lower = donchian_channel(high, low, 20)
    
    # Volume confirmation (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20  # Require 1.5x average volume
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(adx_14_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper + strong trend (ADX>25) + volume surge
            if close[i] > donch_upper[i] and adx_14_1w_aligned[i] > 25 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + strong trend (ADX>25) + volume surge
            elif close[i] < donch_lower[i] and adx_14_1w_aligned[i] > 25 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Donchian channel or trend weakens
            donch_middle = (donch_upper[i] + donch_lower[i]) / 2
            if position == 1:
                if close[i] < donch_middle or adx_14_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_middle or adx_14_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1wADX_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0