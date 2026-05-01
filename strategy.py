#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w ADX > 25 regime filter
# Uses daily Donchian channels for swing trading with weekly trend confirmation
# Volume spike > 2.0x 50-period EMA on weekly timeframe reduces false breakouts
# 1w ADX > 25 ensures strong trending market, avoiding chop and sideways markets
# Designed for low trade frequency (~10-20 trades/year) to minimize fee drag
# Works in bull/bear: ADX filter avoids weak trends, volume confirms institutional participation

name = "1d_Donchian20_1wVolume_1wADX_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for volume and regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_rolling.values
    donchian_lower = low_rolling.values
    
    # 1w volume spike filter: volume > 2.0 * 50-period EMA
    vol_1w = df_1w['volume'].values
    vol_series = pd.Series(vol_1w)
    vol_ema_50 = vol_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_spike_1w = vol_1w > (2.0 * vol_ema_50)
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
    start_idx = max(70, 20)  # Need Donchian, ADX and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in strong trending markets (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if strong_trend:
                # Long: Break above Donchian upper with weekly volume spike
                if close[i] > donchian_upper[i] and volume_spike_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Donchian lower with weekly volume spike
                elif close[i] < donchian_lower[i] and volume_spike_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid weak/choppy markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Donchian lower or opposite breakout
            if close[i] <= donchian_lower[i] or (close[i] < donchian_lower[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Donchian upper or opposite breakout
            if close[i] >= donchian_upper[i] or (close[i] > donchian_upper[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals