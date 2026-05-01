#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d ADX trend filter + volume spike confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 AND volume > 2.0x 20-bar average.
# Short when Bear Power < 0 AND Bull Power < 0 AND 1d ADX > 25 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture medium-term trends with low trade frequency.
# Works in bull (buy strength in uptrend) and bear (sell weakness in downtrend) via 1d ADX trend filter.

name = "6h_ElderRay_1dADX_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA13 and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d_adx[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_adx[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    tr_smooth[tr_period-1] = np.nansum(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.nansum(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.nansum(dm_minus[:tr_period])
    
    for i in range(tr_period, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / tr_period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, (dm_plus_smooth / tr_smooth) * 100, 0)
    di_minus = np.where(tr_smooth != 0, (dm_minus_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_period = 14
    adx = np.zeros_like(dx)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.nanmean(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1d indicators to 6h
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_aligned
    bear_power = low - ema_13_aligned
    
    # Volume confirmation: current 6h volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Elder Ray signals
        bull_strong = bull_power[i] > 0
        bear_weak = bear_power[i] < 0
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume confirmation
            if (bull_strong and 
                bear_weak and 
                adx_aligned[i] > 25 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 AND ADX > 25 AND volume confirmation
            elif (not bull_strong and 
                  bear_weak and 
                  adx_aligned[i] > 25 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 OR ADX falls below 20 (trend weakening)
            if (bull_power[i] <= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR ADX falls below 20 (trend weakening)
            if (bear_power[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals