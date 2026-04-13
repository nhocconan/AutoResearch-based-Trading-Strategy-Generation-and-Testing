#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot + 1d volume spike + 12h ADX regime filter
    # Long: price breaks above Camarilla H3 (1d) + 1d volume > 1.5x 20-period avg + 12h ADX > 20
    # Short: price breaks below Camarilla L3 (1d) + 1d volume > 1.5x 20-period avg + 12h ADX > 20
    # Exit: price returns to Camarilla Pivot (1d) OR 12h ADX < 15 (regime shift to ranging)
    # Uses 6h for entry timing, 1d for Camarilla pivots and volume, 12h for ADX regime
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for ADX (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = C + 1.1*(H-L)*1.1/2, H3 = C + 1.1*(H-L)*1.1/4, L3 = C - 1.1*(H-L)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    camarilla_range = prev_high_1d - prev_low_1d
    h3_1d = prev_close_1d + 1.1 * camarilla_range * 1.1 / 4
    l3_1d = prev_close_1d - 1.1 * camarilla_range * 1.1 / 4
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    
    # Align 1d Camarilla levels to 6h (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate 1d volume spike (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate 12h ADX (Average Directional Index)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing for TR, DM+, DM-
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_12h, np.nan)
    di_minus = np.full_like(atr_12h, np.nan)
    mask = ~np.isnan(atr_12h) & (atr_12h > 0)
    di_plus[mask] = 100 * dm_plus_smoothed[mask] / atr_12h[mask]
    di_minus[mask] = 100 * dm_minus_smoothed[mask] / atr_12h[mask]
    
    # DX and ADX
    dx = np.full_like(atr_12h, np.nan)
    dx_mask = mask & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) > 0)
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    adx_12h = wilders_smoothing(dx, 14)
    
    # Align 12h ADX to 6h (wait for completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h ADX > 20 (trending regime)
        trending_regime = adx_aligned[i] > 20
        # Exit regime: ADX < 15 (ranging/weak trend)
        ranging_regime = adx_aligned[i] < 15
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > h3_aligned[i]) and vol_confirm and trending_regime
        short_entry = (close[i] < l3_aligned[i]) and vol_confirm and trending_regime
        
        # Exit logic: return to pivot OR regime shift to ranging
        long_exit = (close[i] <= pivot_aligned[i]) or ranging_regime
        short_exit = (close[i] >= pivot_aligned[i]) or ranging_regime
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_12h_camarilla_volume_adx_v1"
timeframe = "6h"
leverage = 1.0