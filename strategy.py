#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Channel Breakout with Volume and ADX Filter
# Hypothesis: Price breaking out of 20-period Donchian channel with volume confirmation and
# trending conditions (ADX > 25) captures institutional moves. Works in bull/bear by
# going long on upper breakouts and short on lower breakouts. Uses 1-day trend filter
# to avoid counter-trend trades. Target: 15-25 trades/year (60-100 over 4 years).

name = "12h_donchian20_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily close for trend filter
    daily_close = df_daily['close'].values
    daily_close_series = pd.Series(daily_close)
    daily_ema50 = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Calculate 20-period Donchian channels on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14) for trend strength
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    dm_plus = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0)
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/14)
    def Wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = Wilder_smoothing(tr, 14)
    dm_plus_smooth = Wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = Wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = Wilder_smoothing(dx, 14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian or trend turns bearish
            if close[i] <= donchian_low[i] or ema50_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian or trend turns bullish
            if close[i] >= donchian_high[i] or ema50_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above upper Donchian with volume, ADX > 25, and price above daily EMA50
            if (high[i] > donchian_high[i] and close[i] > donchian_high[i] and 
                vol_filter[i] and adx[i] > 25 and close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian with volume, ADX > 25, and price below daily EMA50
            elif (low[i] < donchian_low[i] and close[i] < donchian_low[i] and 
                  vol_filter[i] and adx[i] > 25 and close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals