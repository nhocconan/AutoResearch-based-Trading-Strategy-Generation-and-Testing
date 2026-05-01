#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h volume spike confirmation and 1d ADX > 25 regime filter
# Uses tighter Camarilla levels (H3/L3) for higher-probability intraday breakouts
# Volume spike > 2.0x 20-period EMA on 4h reduces false breakouts
# 1d ADX > 25 ensures trending market regime to avoid chop
# Session filter (08-20 UTC) avoids low-liquidity hours
# Position size 0.20 to control drawdown and enable multiple concurrent signals
# Target: 15-35 trades/year per symbol (60-140 over 4 years) to minimize fee drag

name = "1h_Camarilla_H3L3_Breakout_4hVolume_1dADX_Regime_v1"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d HTF data for Camarilla levels and ADX regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H3 = close + 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/6
    camarilla_H4_1d = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_L4_1d = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_H3 = camarilla_L4_1d + (camarilla_H4_1d - camarilla_L4_1d) * 3 / 8
    camarilla_L3 = camarilla_H4_1d - (camarilla_H4_1d - camarilla_L4_1d) * 3 / 8
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3.values)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3.values)
    
    # 4h volume spike filter: volume > 2.0 * 20-period EMA
    vol_4h = df_4h['volume'].values
    vol_series_4h = pd.Series(vol_4h)
    vol_ema_20_4h = vol_series_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_4h = vol_4h > (2.0 * vol_ema_20_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
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
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Break above Camarilla H3 with 4h volume spike
                if close[i] > camarilla_H3_aligned[i] and volume_spike_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: Break below Camarilla L3 with 4h volume spike
                elif close[i] < camarilla_L3_aligned[i] and volume_spike_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla L3 or opposite breakout with volume
            if close[i] <= camarilla_L3_aligned[i] or (close[i] < camarilla_L3_aligned[i] and volume_spike_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla H3 or opposite breakout with volume
            if close[i] >= camarilla_H3_aligned[i] or (close[i] > camarilla_H3_aligned[i] and volume_spike_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals