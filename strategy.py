#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume Confirmation and ADX Filter
# Hypothesis: Price breaking out of daily Donchian channels (20-period high/low) 
# with volume confirmation (>1.5x average) and trend filter (ADX > 25) captures
# strong momentum moves while avoiding choppy markets. Works in bull (buy breakouts) 
# and bear (sell breakdowns) by using directional filters. Target: 20-40 trades/year.

name = "4h_daily_donchian_breakout_volume_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    donchian_high = daily_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = daily_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Calculate ADX (14-period) for trend strength
    # +DM and -DM calculation
    high_diff = np.diff(daily_high, prepend=daily_high[0])
    low_diff = np.diff(daily_low, prepend=daily_low[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range calculation
    tr1 = np.abs(np.diff(daily_high, prepend=daily_high[0]))
    tr2 = np.abs(np.diff(daily_low, prepend=daily_low[0]))
    tr3 = np.abs(daily_high - daily_low)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period]) if np.sum(~np.isnan(data[:period])) >= period else 0
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / (atr + 1e-10)
    minus_di = 100 * wilder_smooth(minus_dm, 14) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, 14)
    
    # Shift ADX by 1 to use only completed daily bars
    adx = np.roll(adx, 1)
    if len(adx) > 1:
        adx[0] = adx[1]
    else:
        adx[0] = 0
    
    # Align daily data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below daily Donchian low or trend weakens (ADX < 20)
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above daily Donchian high or trend weakens (ADX < 20)
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above daily Donchian high with volume and strong trend (ADX > 25)
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                adx_aligned[i] > 25 and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below daily Donchian low with volume and strong trend (ADX > 25)
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  adx_aligned[i] > 25 and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals