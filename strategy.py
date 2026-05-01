#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
# 1d ADX(14) > 25 filters for trending markets, < 20 for ranging markets
# In trending markets (ADX>25): fade extreme %R readings (counter-trend pullbacks)
# In ranging markets (ADX<20): fade %R extremes at support/resistance
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~12-37 trades/year per symbol with 0.25 sizing
# Works in both bull and bear markets by adapting to regime (trend vs range)

name = "6h_WilliamsR_1dADX_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    if len(tr) > tr_period:
        tr_smooth[tr_period] = np.nansum(tr[1:tr_period+1])
        dm_plus_smooth[tr_period] = np.nansum(dm_plus[1:tr_period+1])
        dm_minus_smooth[tr_period] = np.nansum(dm_minus[1:tr_period+1])
        
        # Wilder's smoothing
        for i in range(tr_period+1, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/tr_period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1]/tr_period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1]/tr_period) + dm_minus[i]
    
    # Directional Indicators
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    
    # ADX smoothing (period=14)
    if len(dx) > tr_period:
        adx[tr_period] = np.nanmean(dx[1:tr_period+1])
        for i in range(tr_period+1, len(dx)):
            adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d ADX (15 days) + Williams %R (14 periods) + volume EMA20
    start_idx = max(15, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Regime determination
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:  # Flat - look for new entries
            if is_trending:
                # In trending markets: fade extreme Williams %R (pullback entries)
                if wr < -80 and vol_spike:  # Oversold - look for long
                    signals[i] = 0.25
                    position = 1
                elif wr > -20 and vol_spike:  # Overbought - look for short
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # In ranging markets: fade Williams %R extremes at support/resistance
                if wr < -80 and vol_spike:  # Oversold - long
                    signals[i] = 0.25
                    position = 1
                elif wr > -20 and vol_spike:  # Overbought - short
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX 20-25): no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to overbought or ADX weakens
            if wr > -20 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to oversold or ADX weakens
            if wr < -80 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals