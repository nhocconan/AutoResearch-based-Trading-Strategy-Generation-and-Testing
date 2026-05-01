#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w ADX regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: long when %R < -80 (oversold),
# short when %R > -20 (overbought). 1w ADX > 25 filters for trending markets only.
# Volume confirmation (> 1.5x 20-period EMA) ensures institutional participation.
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing.
# Works in bull markets by buying oversold dips in uptrends, and in bear markets
# by selling overbought rallies in downtrends. Avoids ranging markets via ADX filter.

name = "6h_WilliamsR_1wADX_Regime_Volume_v1"
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
    
    # 1w HTF data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 30):
        return np.zeros(n)
    
    # Calculate 1w ADX for regime filter (trending market filter)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]),
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]),
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initialize first values
    tr_smooth[period] = np.nansum(tr[1:period+1])
    dm_plus_smooth[period] = np.nansum(dm_plus[1:period+1])
    dm_minus_smooth[period] = np.nansum(dm_minus[1:period+1])
    
    # Wilder smoothing
    for i in range(period+1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.zeros_like(dx)
    adx[2*period] = np.nanmean(dx[period:2*period+1])
    for i in range(2*period+1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 1w ADX to 6h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 6h Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1w ADX (28+14=42), 6h Williams %R (14), volume EMA (20)
    start_idx = max(42, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1w ADX > 25 (trending market)
        trending = adx_1w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Long: Williams %R oversold (< -80) with volume spike in uptrend
                if williams_r[i] < -80 and volume_spike[i] and close[i] > close[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought (> -20) with volume spike in downtrend
                elif williams_r[i] > -20 and volume_spike[i] and close[i] < close[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (momentum fading) or volume dies
            if williams_r[i] > -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (momentum fading) or volume dies
            if williams_r[i] < -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals