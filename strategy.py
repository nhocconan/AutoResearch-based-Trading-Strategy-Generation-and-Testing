#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d volume spike + 1d ADX regime filter
# Camarilla pivot levels provide high-probability reversal/breakout zones in crypto
# Volume spike confirms institutional participation, reducing false breakouts
# 1d ADX > 20 filters for trending regimes, avoiding whipsaws in ranging markets
# Designed for low frequency (50-150 trades over 4 years) with discrete sizing
# Works in both bull and bear: ADX regime filter avoids ranging markets, volume confirms legitimacy

name = "12h_Camarilla_R3S3_Breakout_1dVolume_1dADX_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d HTF data for volume average and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = np.zeros(len(prices))
    camarilla_s3 = np.zeros(len(prices))
    
    # Shift 1d data by one bar to use previous day's OHLC (no look-ahead)
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 1d volume spike filter: volume > 1.8 * 20-period EMA
    vol_series_1d = pd.Series(volume_1d)
    vol_ema_20_1d = vol_series_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * vol_ema_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d ADX(14) for regime filter
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need ADX and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 20)
        trending = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Break above Camarilla R3 with volume spike
                if close[i] > camarilla_r3_aligned[i] and volume_spike_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Camarilla S3 with volume spike
                elif close[i] < camarilla_s3_aligned[i] and volume_spike_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla S3 or opposite breakout
            if close[i] <= camarilla_s3_aligned[i] or (close[i] < camarilla_s3_aligned[i] and volume_spike_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla R3 or opposite breakout
            if close[i] >= camarilla_r3_aligned[i] or (close[i] > camarilla_r3_aligned[i] and volume_spike_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals