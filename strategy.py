#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high + 1w ADX > 25 + volume > 1.5x 20-day avg volume
# Short when price breaks below 20-day Donchian low + 1w ADX > 25 + volume > 1.5x 20-day avg volume
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (7-25/year).
# Donchian channels provide clear structure, 1w ADX ensures we only trade strong trends avoiding chop,
# volume confirmation adds conviction to breakouts. Works in both bull (breakouts up) and bear (breakdowns down).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian high and low (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
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
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Avoid division by zero
    plus_di = np.where(atr_1w != 0, 100 * plus_dm_smooth / atr_1w, 0)
    minus_di = np.where(atr_1w != 0, 100 * minus_dm_smooth / atr_1w, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Wilder's smoothing for ADX
    adx_1w = np.zeros_like(dx)
    adx_1w[2*period-1] = np.mean(dx[period-1:2*period])
    for i in range(2*period, len(dx)):
        adx_1w[i] = (adx_1w[i-1] * (period-1) + dx[i]) / period
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-day Donchian high
        # 2. Strong trend (1w ADX > 25)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i]) and \
           (adx_1w_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-day Donchian low
        # 2. Strong trend (1w ADX > 25)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i]) and \
             (adx_1w_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_Volume_1wADX25_Filter_v1"
timeframe = "1d"
leverage = 1.0