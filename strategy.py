#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Donchian lower (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above 4h Donchian middle (20) OR volume drops below average
# Uses discrete sizing 0.20 to minimize fee churn. Session filter 08-20 UTC to reduce noise.
# Target: 80-180 total trades over 4 years (20-45/year) for 1h timeframe.
# 4h Donchian provides robust structure from higher timeframe.
# 1d ADX > 25 filters for trending markets only, avoiding choppy/range-bound conditions.
# Volume confirmation ensures breakout validity and reduces false signals.
# Works in bull markets (breakouts with strong uptrend) and bear markets (breakdowns with strong downtrend).

name = "1h_Donchian20_1dADX25_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channel (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (upper + lower)/2
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * np.zeros_like(high)
        minus_di = 100 * np.zeros_like(high)
        dx = np.zeros_like(high)
        
        # Smooth +DM and -DM
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        plus_dm_smooth[period] = np.mean(plus_dm[:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[:period+1])
        for i in range(period+1, len(plus_dm)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        
        # Smooth DX to get ADX
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper, 1d ADX > 25, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian lower, 1d ADX > 25, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian middle OR volume drops below average
            if close[i] < donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian middle OR volume drops below average
            if close[i] > donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals