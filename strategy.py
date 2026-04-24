#!/usr/bin/env python3
"""
Hypothesis: 1h ADX(14) Trend Strength + 4h Donchian(20) Breakout + Volume Spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h Donchian(20) for trend direction (breakout above/below 20-period channel).
- Entry: Long when price breaks above 4h Donchian upper band AND 1h ADX > 25 AND 1h volume > 1.5 * 1h volume MA(20);
         Short when price breaks below 4h Donchian lower band AND 1h ADX > 25 AND 1h volume > 1.5 * 1h volume MA(20).
- Exit: Long exits when price crosses below 4h Donchian middle band; Short exits when price crosses above 4h Donchian middle band.
- Signal size: 0.20 discrete to minimize fee churn.
- Session filter: 08:00-20:00 UTC to avoid low-volume off-hours noise.
- ADX filters for trending markets only, reducing whipsaws in chop.
- Donchian breakouts capture momentum; volume spike confirms conviction.
- Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian channels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate 1h ADX(14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    divisor = np.where(tr14 == 0, 1, tr14)
    plus_di14 = 100 * plus_dm14 / divisor
    minus_di14 = 100 * minus_dm14 / divisor
    dx = np.where((plus_di14 + minus_di14) == 0, 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14))
    adx = wilder_smooth(dx, 14)
    
    # Calculate 1h volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 14+14)  # Donchian needs 20, volume MA needs 20, ADX needs ~28
    
    # Precompute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_adx = adx[i]
        curr_volume = volume[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = curr_adx > 25
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if strong_trend and vol_confirm:
                # Long: price breaks above 4h Donchian upper band
                if curr_close > donchian_high_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below 4h Donchian lower band
                elif curr_close < donchian_low_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below 4h Donchian middle band
            if curr_close < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price crosses above 4h Donchian middle band
            if curr_close > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_ADX_DonchianBreakout_Volume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0