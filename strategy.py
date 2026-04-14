#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ADX trend filter and volume confirmation
# Designed to work in both bull and bear markets by filtering for strong trends (ADX > 25)
# Uses volume spike (2x 20-period average) to confirm breakout strength
# Position size: 0.25 to balance return and drawdown control
# Target: 20-50 trades per year by requiring multiple confluence factors

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data once for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 12h data (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe (waits for completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Donchian channels (20-period) on 4h data
    def donchian_channel(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Volume moving average (20-period) for spike detection
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    for i in range(20, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Volume spike (2x average volume)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Strong trend filter (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Long: Price breaks above Donchian upper with volume spike and strong trend
        if position == 0 and close[i] > donchian_upper[i] and volume_spike and strong_trend:
            position = 1
            signals[i] = position_size
        # Short: Price breaks below Donchian lower with volume spike and strong trend
        elif position == 0 and close[i] < donchian_lower[i] and volume_spike and strong_trend:
            position = -1
            signals[i] = -position_size
        # Exit: Price returns to middle of Donchian channel or opposite breakout
        elif position != 0:
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if position == 1 and close[i] < donchian_middle:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donchian_middle:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0