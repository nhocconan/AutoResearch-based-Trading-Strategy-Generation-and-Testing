#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX regime filter.
# Long when: price breaks above Donchian(20) high AND volume > 1.5x 20-period average volume AND 1d ADX > 25 (trending market).
# Short when: price breaks below Donchian(20) low AND volume > 1.5x 20-period average volume AND 1d ADX > 25.
# Exit when: price crosses Donchian(20) midpoint OR ADX drops below 20 (range market).
# Uses discrete sizing 0.30 to balance return and drawdown. Target: 20-40 trades/year.
# Donchian channels provide structural breakout signals, volume confirms conviction, ADX ensures we trade only in trending regimes.
# Works in bull (breakouts up) and bear (breakouts down) by capturing directional momentum with regime filter.

name = "4h_DonchianBreakout_Volume_ADXRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilder_smooth(values, period):
        """Wilder's smoothing (similar to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        smoothed = np.full_like(values, np.nan)
        smoothed[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
            else:
                smoothed[i] = np.nan
        return smoothed
    
    period_adx = 14
    atr = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = wilder_smooth(dx, period_adx)
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, lookback-1, 19)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_ratio = volume_ratio[i]
        curr_adx = adx_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND volume confirmation AND trending regime (ADX > 25)
            if (curr_close > donchian_high[i] and 
                curr_volume_ratio > 1.5 and 
                curr_adx > 25):
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian low AND volume confirmation AND trending regime (ADX > 25)
            elif (curr_close < donchian_low[i] and 
                  curr_volume_ratio > 1.5 and 
                  curr_adx > 25):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses Donchian midpoint OR ADX drops below 20 (range market)
            if (curr_close < donchian_mid[i] or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price crosses Donchian midpoint OR ADX drops below 20 (range market)
            if (curr_close > donchian_mid[i] or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals