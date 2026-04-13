#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1-day ADX regime filter and volume confirmation.
# Elder Ray measures bull/bear power (bull = high - EMA13, bear = low - EMA13).
# In strong trends, bull/bear power persists; in ranges, it oscillates near zero.
# Combined with 1d ADX (>25 = trending, <20 = ranging) and volume spikes,
# it filters false signals in low-volatility environments.
# Target: 12-37 trades per year (50-150 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM- (14-period)
    def smooth_series(data, period):
        smoothed = np.full(len(data), np.nan)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_series(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray on 6h timeframe (EMA13)
    def ema(data, period):
        ema_vals = np.full(len(data), np.nan)
        if len(data) < period:
            return ema_vals
        multiplier = 2 / (period + 1)
        ema_vals[0] = data[0]
        for i in range(1, len(data)):
            ema_vals[i] = (data[i] - ema_vals[i-1]) * multiplier + ema_vals[i-1]
        return ema_vals
    
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Average volume (20-period = 20 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Strong bull power + trending regime + volume confirmation
            if (bull_val > 0 and is_trending and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Strong bear power + trending regime + volume confirmation
            elif (bear_val < 0 and is_trending and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull power turns negative OR regime shifts to ranging
            if (bull_val <= 0 or not is_trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear power turns positive OR regime shifts to ranging
            if (bear_val >= 0 or not is_trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_ADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0