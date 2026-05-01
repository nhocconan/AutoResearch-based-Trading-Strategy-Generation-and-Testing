#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and 1d ADX > 25 regime filter
# Uses tighter Camarilla levels (H3/L3) for higher probability breakouts
# Volume spike > 2.0x 20-period EMA reduces false breakouts
# 1d ADX > 25 ensures trending market, avoiding whipsaws in ranging conditions
# Designed for optimal trade frequency: ~15-30 trades/year per symbol with 0.25 sizing
# Works in bull/bear: ADX filter avoids false signals in ranging markets, volume confirms institutional participation

name = "12h_Camarilla_H3L3_Breakout_1dVolume_1dADX_Regime_v1"
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
    
    # 1d HTF data for Camarilla levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H3 = close + 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/6
    camarilla_H3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 6
    camarilla_L3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 6
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3.values)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3.values)
    
    # 1d volume spike filter: volume > 2.0 * 20-period EMA (stricter for fewer trades)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    # 1d ADX(14) for regime filter
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
    start_idx = max(30, 20)  # Need ADX and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Break above Camarilla H3 with volume spike
                if close[i] > camarilla_H3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Camarilla L3 with volume spike
                elif close[i] < camarilla_L3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla L3 or opposite breakout
            if close[i] <= camarilla_L3_aligned[i] or (close[i] < camarilla_L3_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla H3 or opposite breakout
            if close[i] >= camarilla_H3_aligned[i] or (close[i] > camarilla_H3_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals