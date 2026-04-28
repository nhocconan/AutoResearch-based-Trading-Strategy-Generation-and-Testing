#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
# Enter long when price breaks above 20-period Donchian high, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Enter short when price breaks below 20-period Donchian low, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Exit when price reaches opposite Donchian level or crosses 12h EMA50.
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining profitability.
# Target: 80-120 total trades over 4 years (20-30/year) to stay within optimal range.
# Donchian channels provide clear breakout levels; 1d ADX ensures we only trade in strong trends (works in bull/bear);
# Volume confirmation filters for institutional participation.

name = "12h_DonchianBreakout_1dADX25_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def ma_smooth(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + (arr[i] / period)
        return result
    
    tr_ma = ma_smooth(tr, 14)
    dm_plus_ma = ma_smooth(dm_plus, 14)
    dm_minus_ma = ma_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_ma != 0, (dm_plus_ma / tr_ma) * 100, 0)
    di_minus = np.where(tr_ma != 0, (dm_minus_ma / tr_ma) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = ma_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian(20)
    donchian_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 12h EMA50 for exit
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 50, 20)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(ema_50[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d ADX trend filter: >25 indicates trending market
        adx_trend = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Exit conditions: price reaches opposite Donchian level or crosses 12h EMA50
        exit_long = close[i] < donchian_low[i] or close[i] < ema_50[i]
        exit_short = close[i] > donchian_high[i] or close[i] > ema_50[i]
        
        # Handle entries and exits
        if breakout_up and adx_trend and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and adx_trend and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals