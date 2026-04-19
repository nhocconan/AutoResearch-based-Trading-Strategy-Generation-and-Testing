#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + 1d volume confirmation + 1w ADX trend strength filter
# - KAMA (Kaufman Adaptive Moving Average) adapts to market noise, effective in both trending and ranging markets
# - Long when price > KAMA, short when price < KAMA, only with 1d volume > 1.5x 20-period average
# - 1w ADX > 25 ensures we only trade in strong trends (avoids choppy markets where trend following fails)
# - Designed to capture trending moves while avoiding false signals in low volatility/choppy conditions
# - Target: 25-40 trades/year to minimize fee drag

name = "4h_KAMA_1dVolume_1wADXTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for ADX trend strength
    df_1w = get_htf_data(prices, '1w')
    
    # 1w ADX calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_series(data, period):
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    atr_1w = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w > 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w > 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = smooth_series(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # KAMA calculation (10-period ER, 2/30 SC)
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 9 values
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    
    # Efficiency Ratio
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing Constants
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)  # Fast=2, Slow=30
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama[i]) or np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        # Trend strength filter: 1w ADX > 25
        trend_filter = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Look for long entry: price > KAMA + volume + trend strength
            if close[i] > kama[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price < KAMA + volume + trend strength
            elif close[i] < kama[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price < KAMA or trend weakens
            if close[i] < kama[i] or adx_1w_aligned[i] < 20:  # Exit if trend weakens (ADX < 20)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price > KAMA or trend weakens
            if close[i] > kama[i] or adx_1w_aligned[i] < 20:  # Exit if trend weakens (ADX < 20)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals