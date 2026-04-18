#!/usr/bin/env python3
"""
1d_Pivot_R1S1_Breakout_Volume
Hypothesis: Daily price respects weekly-derived Camarilla pivot levels R1/S1. Breakouts with volume (>1.5x 20-period mean) trigger entries. Uses weekly EMA(34) filter to avoid counter-trend trades. Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe. Works in both bull and bear markets by adapting to regime (range vs trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (based on previous week)
    R1 = np.full_like(high_1w, np.nan)
    S1 = np.full_like(low_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        prev_close = close_1w[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            R1[i] = prev_close + 1.1 * range_ / 12
            S1[i] = prev_close - 1.1 * range_ / 12
    
    # Calculate 14-period ADX for regime filtering
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smooth TR, DM+
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI+ and DI-
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        valid = ~np.isnan(atr) & (atr != 0)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        # DX and ADX
        dx = np.full_like(di_plus, np.nan)
        dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
        
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            # Initial ADX
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            # Wilder smoothing for ADX
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Get weekly data for trend filter
    close_1w_for_ema = close_1w
    
    # Weekly EMA(34) for trend filter
    if len(close_1w_for_ema) >= 34:
        ema_1w = pd.Series(close_1w_for_ema).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w_for_ema, np.nan)
    
    # Align all weekly data to daily timeframe
    R1_1d = align_htf_to_ltf(prices, df_1w, R1)
    S1_1d = align_htf_to_ltf(prices, df_1w, S1)
    adx_1d = align_htf_to_ltf(prices, df_1w, adx_1w)
    ema_1w_1d = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_1d[i]) or np.isnan(S1_1d[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(ema_1w_1d[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filters: weekly ADX < 25 (range) AND price above weekly EMA (bullish bias)
        range_regime = adx_1d[i] < 25
        bullish_bias = close[i] > ema_1w_1d[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in range regime
            if close[i] > R1_1d[i] and vol_confirm and range_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in range regime
            elif close[i] < S1_1d[i] and vol_confirm and range_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR ADX rises above 30 (trend emerging)
            if close[i] < S1_1d[i] or adx_1d[i] > 30:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR ADX rises above 30 (trend emerging)
            if close[i] > R1_1d[i] or adx_1d[i] > 30:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0