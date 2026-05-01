#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w volume confirmation and 1w ADX > 20 regime filter
# Uses Camarilla H3/L3 levels (inner levels) for higher-probability breakouts
# 1w volume spike > 2.0x 20-period EMA reduces false breakouts
# 1w ADX > 20 ensures trending market regime (avoids ranging markets)
# Designed for optimal trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Works in bull/bear: ADX filter avoids strong ranging markets, volume confirms participation

name = "1d_Camarilla_H3L3_Breakout_1wVolume_1wADX_Regime_v1"
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
    
    # 1w HTF data for Camarilla levels, volume, and regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1w bar
    # H3 = close + 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/6
    camarilla_H3 = df_1w['close'] + 1.1 * (df_1w['high'] - df_1w['low']) / 6
    camarilla_L3 = df_1w['close'] - 1.1 * (df_1w['high'] - df_1w['low']) / 6
    
    # Align Camarilla levels to 1d timeframe (wait for 1w bar to close)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H3.values)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L3.values)
    
    # 1w volume spike filter: volume > 2.0 * 20-period EMA (tight for optimal frequency)
    vol_series = pd.Series(df_1w['volume'].values)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1w = df_1w['volume'].values > (2.0 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
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
    start_idx = max(27, 20)  # Need ADX and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 20)
        trending = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Break above Camarilla H3 with volume spike
                if close[i] > camarilla_H3_aligned[i] and volume_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Camarilla L3 with volume spike
                elif close[i] < camarilla_L3_aligned[i] and volume_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla L3 or opposite breakout
            if close[i] <= camarilla_L3_aligned[i] or (close[i] < camarilla_L3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla H3 or opposite breakout
            if close[i] >= camarilla_H3_aligned[i] or (close[i] > camarilla_H3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals