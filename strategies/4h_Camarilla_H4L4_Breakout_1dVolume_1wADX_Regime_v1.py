#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 1d volume confirmation and 1w ADX > 20 regime filter
# Uses inner Camarilla levels (H4/L4) for tighter, higher-probability breakouts
# Volume spike > 1.8x 20-period EMA reduces false breakouts
# 1w ADX > 20 ensures trending market regime (more permissive than ADX>25 for more trades)
# Designed for optimal trade frequency: ~30-50 trades/year per symbol with 0.30 sizing
# Works in bull/bear: ADX filter avoids strong ranging markets, volume confirms participation

name = "4h_Camarilla_H4L4_Breakout_1dVolume_1wADX_Regime_v1"
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
    
    # 1d HTF data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w HTF data for regime filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.1*(high - low)/4
    # L4 = close - 1.1*(high - low)/4
    camarilla_H4 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_L4 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4.values)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4.values)
    
    # 1d volume spike filter: volume > 1.8 * 20-period EMA (balanced for trade frequency)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    # 1w ADX(14) for regime filter (using standard period)
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
    start_idx = max(27, 20)  # Need ADX and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 20)
        trending = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Break above Camarilla H4 with volume spike
                if close[i] > camarilla_H4_aligned[i] and volume_spike[i]:
                    signals[i] = 0.30
                    position = 1
                # Short: Break below Camarilla L4 with volume spike
                elif close[i] < camarilla_L4_aligned[i] and volume_spike[i]:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla L4 or opposite breakout
            if close[i] <= camarilla_L4_aligned[i] or (close[i] < camarilla_L4_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla H4 or opposite breakout
            if close[i] >= camarilla_H4_aligned[i] or (close[i] > camarilla_H4_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals