#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w ADX > 25 regime filter
# Uses Donchian channel breakouts for trend capture with volume confirmation to reduce false signals
# 1w ADX > 25 ensures we only trade in strong trending markets (avoids chop/range)
# Designed for low trade frequency (~15-25 trades/year) with 0.30 sizing to minimize fee drag
# Works in bull/bear markets: ADX filter avoids ranging conditions, volume confirms institutional participation

name = "1d_Donchian20_1wVolume_1wADX_Regime_v1"
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
    
    # 1d data for Donchian channels (primary timeframe)
    df_1d = prices  # primary timeframe data
    
    # 1w HTF data for volume and regime filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d timeframe
    # Upper channel = highest high over past 20 days
    # Lower channel = lowest low over past 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1w volume confirmation: volume > 1.5 * 20-period EMA
    vol_1w = df_1w['volume'].values
    vol_series_1w = pd.Series(vol_1w)
    vol_ema_20_1w = vol_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1w = vol_1w > (1.5 * vol_ema_20_1w)
    
    # Align 1w volume spike to 1d timeframe
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    # 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
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
    
    # Wilder's smoothing function
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
    
    # Start after warmup (need Donchian 20-period + ADX)
    start_idx = max(30, 20)  # 30 for ADX warmup, 20 for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in strong trending markets (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if strong_trend:
                # Long: Break above Donchian upper channel with volume spike
                if close[i] > donchian_upper[i] and volume_spike_1w_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                # Short: Break below Donchian lower channel with volume spike
                elif close[i] < donchian_lower[i] and volume_spike_1w_aligned[i]:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid weak/choppy markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Donchian lower channel or opposite breakout
            if close[i] <= donchian_lower[i] or (close[i] < donchian_lower[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price returns to Donchian upper channel or opposite breakout
            if close[i] >= donchian_upper[i] or (close[i] > donchian_upper[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals