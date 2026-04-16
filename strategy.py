#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Camarilla R1 level + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S1 level + 1d ADX > 25 + volume confirmation
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# Camarilla pivot levels provide mathematically derived support/resistance that work well in ranging markets.
# ADX filter ensures we only trade in trending conditions, reducing whipsaws in choppy markets.
# Volume confirmation adds conviction to breakouts.
# Target: 20-40 trades/year on 12h timeframe to stay within fee drag limits.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial sums
    tr_sum[tr_period] = np.sum(tr[1:tr_period+1])
    dm_plus_sum[tr_period] = np.sum(dm_plus[1:tr_period+1])
    dm_minus_sum[tr_period] = np.sum(dm_minus[1:tr_period+1])
    
    # Wilder's smoothing
    for i in range(tr_period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI and DX
    di_plus = np.where(tr_sum != 0, 100 * dm_plus_sum / tr_sum, 0)
    di_minus = np.where(tr_sum != 0, 100 * dm_minus_sum / tr_sum, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    adx[2*tr_period] = np.mean(dx[tr_period:2*tr_period+1])
    for i in range(2*tr_period + 1, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h Camarilla Pivot Levels (R1, S1) ===
    # Based on previous period's high, low, close
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r1 = pivot_point + 1.1 * (prev_high - prev_low) / 12.0
    camarilla_s1 = pivot_point - 1.1 * (prev_high - prev_low) / 12.0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # ADX + Camarilla + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Trend filter: 1d ADX > 25 (trending market)
        trend_filter = adx_1d_aligned[i] > 25.0
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 level
        # 2. 1d ADX > 25 (trending)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1[i]) and \
           trend_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 level
        # 2. 1d ADX > 25 (trending)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1[i]) and \
             trend_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dADX_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0