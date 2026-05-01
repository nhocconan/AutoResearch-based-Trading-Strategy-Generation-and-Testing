#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX > 20 regime filter and volume confirmation
# Donchian channels provide robust structure-based breakouts in both bull and bear markets
# ADX > 20 ensures we trade in sufficient momentum environments (avoids chop)
# Volume > 1.3x 20-period EMA confirms participation without being too restrictive
# Designed for optimal trade frequency: ~25-35 trades/year per symbol with 0.30 sizing
# Uses discrete position sizing (0.0, ±0.30) to minimize fee churn from signal changes

name = "4h_Donchian20_Volume_1dADX_Regime_v1"
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
    
    # 1d HTF data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d ADX(14) for regime filter
    # True Range
    tr1 = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[1:])
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > 
                       (df_1d['low'].values[:-1] - df_1d['low'].values[1:]), 
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    dm_minus = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > 
                        (df_1d['high'].values[1:] - df_1d['high'].values[:-1]), 
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.3 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, tr_period + tr_period)  # Need Donchian and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in sufficient momentum environments (ADX > 20)
        sufficient_momentum = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            if sufficient_momentum:
                # Long: Break above Donchian upper band with volume spike
                if close[i] > highest_high[i] and volume_spike[i]:
                    signals[i] = 0.30
                    position = 1
                # Short: Break below Donchian lower band with volume spike
                elif close[i] < lowest_low[i] and volume_spike[i]:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid low momentum/choppy markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Donchian lower band (mean reversion)
            if close[i] <= lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price returns to Donchian upper band (mean reversion)
            if close[i] >= highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals