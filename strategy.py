#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d Volume Spike + 1d ADX Trend Filter
# Uses daily ADX(14) to filter trending markets, daily volume spike (>2x 20-period average) for entry timing,
# and 4h Donchian channel breakout for entry signals. Designed to work in both bull and bear markets
# by only trading in the direction of the daily trend. Target: 20-50 trades/year.

name = "4h_Donchian20_1dADX14_VolumeSpike"
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
    
    # Get daily data for ADX, volume average, and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate True Range (TR)
    tr = np.zeros(len(high_daily))
    tr[0] = high_daily[0] - low_daily[0]
    for i in range(1, len(high_daily)):
        tr[i] = max(high_daily[i] - low_daily[i],
                    abs(high_daily[i] - close_daily[i-1]),
                    abs(low_daily[i] - close_daily[i-1]))
    
    # Calculate Directional Movement (+DM and -DM)
    plus_dm = np.zeros(len(high_daily))
    minus_dm = np.zeros(len(high_daily))
    for i in range(1, len(high_daily)):
        up_move = high_daily[i] - high_daily[i-1]
        down_move = low_daily[i-1] - low_daily[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values use Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = np.zeros(len(atr))
    for i in range(len(dx)):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]) and (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = np.nan
    adx = wilder_smooth(dx, 14)
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(len(high), np.nan)
    donchian_low = np.full(len(low), np.nan)
    if len(high) >= 20:
        for i in range(20, len(high)):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Align daily indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in trending market with volume spike
            # ADX > 25 indicates trending market
            trending_market = adx_aligned[i] > 25
            
            # Long when price breaks above Donchian high in uptrend
            long_condition = (
                close[i] > donchian_high[i] and   # price breaks above Donchian high
                trending_market and               # trending market (ADX > 25)
                vol_spike                         # volume spike for entry
            )
            
            # Short when price breaks below Donchian low in downtrend
            short_condition = (
                close[i] < donchian_low[i] and    # price breaks below Donchian low
                trending_market and               # trending market (ADX > 25)
                vol_spike                         # volume spike for entry
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low or trend weakens
            if close[i] < donchian_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or trend weakens
            if close[i] > donchian_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals