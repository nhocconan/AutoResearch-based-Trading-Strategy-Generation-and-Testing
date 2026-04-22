# 3.14159265358979323846
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Donchian channel breakouts capture breakout moves while ADX > 25 filters for trending conditions.
# Volume confirmation ensures institutional participation. Designed for low trade frequency (~20-35/year)
# to minimize fee decay. Works in both bull and bear markets by following higher timeframe trend.
# Uses 1d ADX to avoid false breakouts in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr14 = wilders_smoothing(tr, period)
    dm_plus14 = wilders_smoothing(dm_plus, period)
    dm_minus14 = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Calculate Donchian channels on 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_6h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_6h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align 1d ADX to 6h timeframe (waits for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = vol > 1.5 * vol_ma
        
        # Trend filter: ADX > 25
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long conditions: price breaks above upper channel + trend + volume
            if price > upper and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel + trend + volume
            elif price < lower and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to middle of channel or trend weakens
            middle = (upper + lower) / 2
            exit_signal = False
            
            if position == 1:  # long position
                if price < middle or adx_val < 20:
                    exit_signal = True
            elif position == -1:  # short position
                if price > middle or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_DonchianBreakout_1dADX_Volume"
timeframe = "6h"
leverage = 1.0