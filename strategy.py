#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and 1w ADX trend filter
# Long when Williams %R(14) < -80 (oversold) + volume > 2.0x 20-period 1d volume avg + 1w ADX > 25 (strong trend)
# Short when Williams %R(14) > -20 (overbought) + volume > 2.0x 20-period 1d volume avg + 1w ADX > 25 (strong trend)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (15-35/year).
# Williams %R identifies exhaustion points in trends; volume spike confirms participation; 1w ADX ensures we trade with the dominant trend.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend) by requiring 1w ADX > 25.

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
    
    # Get 1d HTF data once before loop (for volume SMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop (for ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    vol_sma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1w Indicator: ADX (trend strength filter) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_1w_shift = np.roll(high_1w, 1)
    low_1w_shift = np.roll(low_1w, 1)
    high_1w_shift[0] = high_1w[0]
    low_1w_shift[0] = low_1w[0]
    
    plus_dm = np.where((high_1w - high_1w_shift) > (low_1w_shift - low_1w), 
                       np.maximum(high_1w - high_1w_shift, 0), 0)
    minus_dm = np.where((low_1w_shift - low_1w) > (high_1w - high_1w_shift), 
                        np.maximum(low_1w_shift - low_1w, 0), 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr_1w = np.zeros_like(tr)
    atr_1w[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * (period-1) + tr[i]) / period
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Avoid division by zero
    plus_di_1w = np.where(atr_1w != 0, 100 * plus_dm_smooth / atr_1w, 0)
    minus_di_1w = np.where(atr_1w != 0, 100 * minus_dm_smooth / atr_1w, 0)
    
    dx_1w = np.where((plus_di_1w + minus_di_1w) != 0, 
                     100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 0)
    
    # Wilder's smoothing for ADX
    adx_1w = np.zeros_like(dx_1w)
    adx_1w[2*period-1] = np.mean(dx_1w[period-1:2*period])
    for i in range(2*period, len(dx_1w)):
        adx_1w[i] = (adx_1w[i-1] * (period-1) + dx_1w[i]) / period
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === Williams %R (14-period) on 4h chart ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denom = highest_high_14 - lowest_low_14
    williams_r = np.where(denom != 0, 
                          -100 * (highest_high_14 - close) / denom, 
                          -50)  # neutral when no range
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period 1d volume SMA
        vol_confirm = volume[i] > (vol_sma_20_1d_aligned[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Strong trend (1w ADX > 25)
        # 3. Volume confirmation
        if (williams_r[i] < -80) and \
           (adx_1w_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Strong trend (1w ADX > 25)
        # 3. Volume confirmation
        elif (williams_r[i] > -20) and \
             (adx_1w_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR_VolumeSpike_1wADX25_Filter_v1"
timeframe = "4h"
leverage = 1.0