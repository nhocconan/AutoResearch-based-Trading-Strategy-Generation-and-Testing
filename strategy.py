#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d ADX > 20 regime filter
# Primary timeframe is 1h for entry timing precision; 4h/1d used for signal direction and filtering.
# Camarilla levels provide statistically significant intraday support/resistance.
# Volume spike on 4h confirms breakout legitimacy, reducing false signals.
# 1d ADX > 20 ensures we only trade in trending markets, avoiding chop.
# Designed for moderate frequency: ~15-37 trades/year per symbol (60-150 over 4 years) with discrete sizing.
# Works in bull/bear: ADX filter avoids ranging markets, volume confirms institutional participation.

name = "1h_Camarilla_R3S3_Breakout_4hVolume_1dADX_Regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for Camarilla levels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1d HTF data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_R3 = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low']) / 2
    camarilla_S3 = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low']) / 2
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3.values)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3.values)
    
    # 4h volume spike filter: volume > 2.0 * 20-period EMA
    vol_series_4h = pd.Series(df_4h['volume'].values)
    vol_ema_20_4h = vol_series_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_4h = df_4h['volume'].values > (2.0 * vol_ema_20_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # 1d ADX(20) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
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
    
    tr_period = 20
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need ADX and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 20)
        trending = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Break above Camarilla R3 with volume spike
                if close[i] > camarilla_R3_aligned[i] and volume_spike_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: Break below Camarilla S3 with volume spike
                elif close[i] < camarilla_S3_aligned[i] and volume_spike_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla S3 or opposite breakout
            if close[i] <= camarilla_S3_aligned[i] or (close[i] < camarilla_S3_aligned[i] and volume_spike_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla R3 or opposite breakout
            if close[i] >= camarilla_R3_aligned[i] or (close[i] > camarilla_R3_aligned[i] and volume_spike_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals