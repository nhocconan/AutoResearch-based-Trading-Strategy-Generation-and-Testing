#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w ADX > 25 regime filter
# Uses daily Donchian channels for structure, weekly volume spike for participation,
# and weekly ADX > 25 to ensure trending markets (avoids whipsaws in ranging conditions)
# Designed for low trade frequency: ~10-20 trades/year per symbol with 0.25 sizing
# Works in bull/bear: ADX filter avoids false breakouts in sideways markets,
# volume confirmation ensures institutional participation, Donchian provides clear breakout levels

name = "1d_Donchian20_1wVolume_1wADX25_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for volume and regime filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) from daily data
    # We need to get daily data first for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Daily Donchian(20): upper = 20-day high, lower = 20-day low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels with proper min_periods
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high_1d, 20)
    donchian_lower = rolling_min(low_1d, 20)
    
    # Align Donchian levels to 1d timeframe (wait for daily bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 1w volume spike filter: volume > 2.0 * 20-period EMA (strict for low frequency)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    vol_series = pd.Series(volume_1w)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1w > (2.0 * vol_ema_20)
    
    # 1w ADX(14) for regime filter (using standard period)
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 25)  # Need ADX, Donchian, and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i - (len(prices) - len(df_1w))])):
            signals[i] = 0.0
            continue
        
        # Get volume spike value for current index (need to map 1w index to 1d index)
        # Since we're on 1d timeframe, we need the aligned volume spike
        vol_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike.astype(float))
        
        # Regime filter: only trade in strongly trending markets (ADX > 25)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Break above Donchian upper with volume spike
                if close[i] > donchian_upper_aligned[i] and vol_spike_aligned[i] > 0.5:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Donchian lower with volume spike
                elif close[i] < donchian_lower_aligned[i] and vol_spike_aligned[i] > 0.5:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging/weak trend markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Donchian lower or opposite breakout with volume
            if close[i] <= donchian_lower_aligned[i] or (close[i] < donchian_lower_aligned[i] and vol_spike_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Donchian upper or opposite breakout with volume
            if close[i] >= donchian_upper_aligned[i] or (close[i] > donchian_upper_aligned[i] and vol_spike_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals