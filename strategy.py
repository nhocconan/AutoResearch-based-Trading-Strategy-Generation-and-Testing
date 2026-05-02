#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d EMA34 trend filter, volume confirmation (>1.5x average), and ADX regime filter (ADX > 25)
# Elder Ray measures bull/bear power relative to EMA13 - effective in both trending and ranging markets.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation filters for institutional participation.
# ADX > 25 ensures trades only in trending markets, avoiding whipsaws in ranges.
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years.
# Primary timeframe: 6h, HTF: 1d for EMA34 and ADX.

name = "6h_ElderRay_Power_1dEMA34_ADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND price > 1d EMA34 AND ADX > 25 AND volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND price < 1d EMA34 AND ADX > 25 AND volume spike
            elif (bear_power[i] > 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 OR price < 1d EMA34
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 OR price > 1d EMA34
            if bear_power[i] <= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals