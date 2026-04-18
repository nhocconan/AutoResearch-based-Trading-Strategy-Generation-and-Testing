# 12h_Vortex_Trend_with_Volume_Spike_and_Momentum_Filter
# Hypothesis: Vortex indicator identifies trend direction on 12h timeframe with strong momentum confirmation.
# Uses 1w ADX as regime filter to avoid choppy markets and volume spike for confirmation.
# Designed for low trade frequency (12-37/year) with strong trend capture in both bull and bear markets.
# Combines Vortex crossover signals with momentum (RSI) and volume confirmation for high-probability entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_ata(prices, '1w')
    
    # Calculate ADX on weekly data (14 period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Plus Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    
    # Minus Directional Movement
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values are Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    dm_plus_smooth = wilders_smooth(dm_plus, period)
    dm_minus_smooth = wilders_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, period)
    
    # Align weekly ADX to 12h timeframe (wait for weekly close)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Get daily data for Vortex indicator
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Vortex indicator on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Vortex calculation
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])  # |High - Prior Low|
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])  # |Low - Prior High|
    
    # True Range for Vortex
    tr1_v = np.abs(high_1d[1:] - low_1d[1:])
    tr2_v = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_v = np.abs(low_1d[1:] - close_1d[:-1])
    tr_v = np.maximum(tr1_v, np.maximum(tr2_v, tr3_v))
    tr_v = np.concatenate([[np.nan], tr_v])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    def sum_periods(data, period):
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            if i >= period-1:
                result[i] = np.nansum(data[i-period+1:i+1])
        return result
    
    period_v = 14
    vm_plus_sum = sum_periods(vm_plus, period_v)
    vm_minus_sum = sum_periods(vm_minus, period_v)
    tr_sum = sum_periods(tr_v, period_v)
    
    # VI+ and VI-
    vi_plus = np.where(tr_sum != 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum != 0, vm_minus_sum / tr_sum, 0)
    
    # Align Vortex to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Momentum filter: RSI(14) on 12h data
    def rsi(data, period):
        delta = np.diff(data, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(data, np.nan)
        avg_loss = np.full_like(data, np.nan)
        
        # First average
        if len(data) >= period:
            avg_gain[period-1] = np.nanmean(gain[1:period])
            avg_loss[period-1] = np.nanmean(loss[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(avg_gain[i-1]):
                    avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                if not np.isnan(avg_loss[i-1]):
                    avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_12h = rsi(close, 14)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:  # 20-period lookback
            vol_ma[i] = np.nanmean(volume[i-19:i+1])
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after all indicators are valid
    start_idx = max(50, 20)  # Vortex needs 28 periods (14*2), RSI needs 14, volume needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vi_plus_aligned[i]) or
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(rsi_12h[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1w_aligned[i]
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        rsi_val = rsi_12h[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: Only trade when ADX > 25 (trending market)
        if adx_val < 25:
            # In chop, reduce position or stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ crosses above VI- with RSI > 50 and volume spike
            if vi_plus_val > vi_minus_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: VI- crosses above VI+ with RSI < 50 and volume spike
            elif vi_minus_val > vi_plus_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: VI- crosses above VI+ OR RSI < 40
            if vi_minus_val > vi_plus_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: VI+ crosses above VI- OR RSI > 60
            if vi_plus_val > vi_minus_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Vortex_Trend_with_Volume_Spike_and_Momentum_Filter"
timeframe = "12h"
leverage = 1.0