#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 12h volume spike (1.5x 20-bar avg volume) and 1d ADX regime filter (ADX>25 for trending, ADX<20 for ranging).
# Long when price breaks above Donchian upper channel AND volume spike AND 12h ADX>25 (strong trend).
# Short when price breaks below Donchian lower channel AND volume spike AND 12h ADX>25.
# In ranging markets (12h ADX<20), mean revert at Donchian mid-channel: long when price crosses below mid AND volume spike, short when price crosses above mid AND volume spike.
# Uses discrete position size 0.25 to minimize fee drag. Target: 20-50 trades/year.
# Works in bull/bear via trend filter and ranging via mean reversion filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data once before loop for Donchian channels (using 1d high/low for structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 12h Indicators: ADX(14) for regime filter ===
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def wilders_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value: simple average
            result[period-1] = np.nanmean(data[1:period]) if period > 1 else data[1]
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilders_smooth(tr, period)
        dm_plus_smooth = wilders_smooth(dm_plus, period)
        dm_minus_smooth = wilders_smooth(dm_minus, period)
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # === 4h Indicators: Donchian(20) channels ===
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        mid = (upper + lower) / 2
        return upper, lower, mid
    
    dc_upper, dc_lower, dc_mid = donchian_channels(high, low, 20)
    
    # === Volume spike: 1.5x 20-bar average volume ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Align all indicators to primary timeframe (4h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower)
    dc_mid_aligned = align_htf_to_ltf(prices, df_1d, dc_mid)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # covers Donchian(20) and volume MA(20)
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(dc_upper_aligned[i]) or 
            np.isnan(dc_lower_aligned[i]) or np.isnan(dc_mid_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        adx = adx_aligned[i]
        dc_upper = dc_upper_aligned[i]
        dc_lower = dc_lower_aligned[i]
        dc_mid = dc_mid_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price crosses below Donchian mid-channel
            if price < dc_mid:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian mid-channel
            if price > dc_mid:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trending market (ADX > 25): Donchian breakout with volume spike
            if adx > 25:
                if price > dc_upper and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif price < dc_lower and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX < 20): mean reversion at Donchian mid-channel with volume spike
            elif adx < 20:
                if price < dc_mid and vol_spike and close[i-1] >= dc_mid_aligned[i-1] if i > 0 and not np.isnan(dc_mid_aligned[i-1]) else False:
                    # Price crossed below mid from above
                    signals[i] = 0.25
                    position = 1
                elif price > dc_mid and vol_spike and close[i-1] <= dc_mid_aligned[i-1] if i > 0 and not np.isnan(dc_mid_aligned[i-1]) else False:
                    # Price crossed above mid from below
                    signals[i] = -0.25
                    position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADXRegime_MeanRev_V1"
timeframe = "4h"
leverage = 1.0