#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX(14) trend filter with 4h Donchian(20) breakout and volume spike
# Long when price breaks above 4h Donchian upper band + 4h ADX > 25 (strong trend) + volume > 1.5x 20-period avg + 08-20 UTC session
# Short when price breaks below 4h Donchian lower band + 4h ADX > 25 + volume confirmation + session filter
# Uses discrete position sizing (0.20) to control drawdown and minimize fee drag.
# 4h ADX provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-40 trades/year on 1h timeframe to avoid overtrading.
# 4h Donchian channels provide structure-based breakout levels that work in ranging and trending markets.
# Session filter (08-20 UTC) reduces noise trades during low-volume periods.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: ADX(14) and Donchian(20) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = np.concatenate([[np.nan], high_4h[1:] - high_4h[:-1]])
    down_move = np.concatenate([[np.nan], low_4h[:-1] - low_4h[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period]) if period > 1 else data[0]
            # Wilder's smoothing: result[i] = result[i-1] - (result[i-1]/period) + (data[i]/period)
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
                else:
                    result[i] = result[i-1]
        return result
    
    # Smoothed TR, +DM, -DM
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smoothing(dx, 14)
    
    # Donchian Channel (20-period)
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Align 4h indicators to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Volume SMA for confirmation (using 20-period on 1h)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # ADX(14) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(donchian_upper_4h_aligned[i]) or
            np.isnan(donchian_lower_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_4h_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper band
        # 2. Strong trend (ADX > 25)
        # 3. Volume confirmation
        if (close[i] > donchian_upper_4h_aligned[i]) and \
           strong_trend and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower band
        # 2. Strong trend (ADX > 25)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower_4h_aligned[i]) and \
             strong_trend and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_ADX14_4hDonchian20_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0