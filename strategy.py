#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels with weekly trend filter and volume confirmation
# Long when price breaks above Camarilla H3 level with weekly bullish trend and volume spike
# Short when price breaks below Camarilla L3 level with weekly bearish trend and volume spike
# Exit when price crosses Camarilla pivot (midpoint)
# Uses weekly ADX trend filter to avoid counter-trend trades in choppy markets
# Target: 15-35 trades per symbol over 4 years (4-9/year) to minimize fee drag
# This combines proven Camarilla pivot strategy with trend and volume filters for better robustness

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate daily Camarilla pivot levels
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    pivot = (high_daily + low_daily + close_daily) / 3
    range_daily = high_daily - low_daily
    camarilla_h3 = pivot + (range_daily * 1.1 / 2)
    camarilla_l3 = pivot - (range_daily * 1.1 / 2)
    camarilla_pivot = pivot  # midpoint for exit
    
    # Calculate weekly ADX for trend filter (14-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate True Range
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Calculate Directional Movement
    dm_plus = np.where((high_weekly - np.roll(high_weekly, 1)) > (np.roll(low_weekly, 1) - low_weekly),
                       np.maximum(high_weekly - np.roll(high_weekly, 1), 0), 0)
    dm_minus = np.where((np.roll(low_weekly, 1) - low_weekly) > (high_weekly - np.roll(high_weekly, 1)),
                        np.maximum(np.roll(low_weekly, 1) - low_weekly, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # Calculate DI and DX
    di_plus = np.zeros_like(atr)
    di_minus = np.zeros_like(atr)
    dx = np.zeros_like(atr)
    
    mask = atr != 0
    di_plus[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
    di_minus[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
    dx_mask = (di_plus + di_minus) != 0
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    # Calculate ADX (smoothed DX)
    adx_period = 14
    adx = np.zeros_like(dx)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.mean(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_daily, camarilla_pivot)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # for Camarilla and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_daily_current = volume[i]  # Current daily volume (approximation for 12h)
        
        if position == 0:
            # Long setup: break above Camarilla H3 with weekly trend (ADX > 25) and volume spike
            if (price > camarilla_h3_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_daily_current > 1.5 * vol_ma_daily_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla L3 with weekly trend (ADX > 25) and volume spike
            elif (price < camarilla_l3_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_daily_current > 1.5 * vol_ma_daily_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla pivot
            if price < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla pivot
            if price > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0