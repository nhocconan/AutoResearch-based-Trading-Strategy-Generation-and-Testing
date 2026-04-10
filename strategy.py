#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian upper band AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Short when price breaks below 20-period Donchian lower band AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price crosses back below/above the middle of the Donchian channel (10-period average of upper/lower)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian channels provide clear breakout levels with built-in stoploss via channel width
# - 1d ADX ensures we only trade in trending markets (avoids chop)
# - Volume confirmation reduces false breakouts

name = "4h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20-period)
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    donchian_upper, donchian_lower, donchian_middle = donchian_channels(high, low, 20)
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[1:period])  # First value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(atr_1d != 0, (plus_dm_smooth / atr_1d) * 100, 0)
    minus_di = np.where(atr_1d != 0, (minus_dm_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # ADX trend filter: trending when ADX > 25
    adx_trending = adx > 25
    
    # Align HTF indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), volume_spike)
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper Donchian band AND ADX trending AND volume spike
            if (close[i] > donchian_upper[i] and 
                adx_trending_aligned[i] and 
                volume_spike_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower Donchian band AND ADX trending AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  adx_trending_aligned[i] and 
                  volume_spike_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses back below/above the middle of the Donchian channel
            exit_long = (position == 1 and close[i] < donchian_middle[i])
            exit_short = (position == -1 and close[i] > donchian_middle[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals