#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1w ADX trend filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period volume SMA + 1w ADX > 20
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period volume SMA + 1w ADX > 20
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Donchian channels provide objective trend-following structure. 1w ADX ensures we only trade strong weekly trends.
# Works in bull markets (breakouts continuation) and bear markets (strong downtrend continuations) by requiring ADX > 20.

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
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Donchian Channels (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels: upper = max(high, period), lower = min(low, period)
    period = 20
    donchian_high = np.full_like(high_12h, np.nan)
    donchian_low = np.full_like(low_12h, np.nan)
    
    for i in range(period-1, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-period+1:i+1])
        donchian_low[i] = np.min(low_12h[i-period+1:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 12h Indicator: Volume SMA (20-period) ===
    vol_sma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
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
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr_1w = np.zeros_like(tr)
    atr_1w[period_adx-1] = np.mean(tr[:period_adx])
    for i in range(period_adx, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * (period_adx-1) + tr[i]) / period_adx
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period_adx-1] = np.mean(plus_dm[:period_adx])
    minus_dm_smooth[period_adx-1] = np.mean(minus_dm[:period_adx])
    
    for i in range(period_adx, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period_adx-1) + plus_dm[i]) / period_adx
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period_adx-1) + minus_dm[i]) / period_adx
    
    # Avoid division by zero
    plus_di_1w = np.where(atr_1w != 0, 100 * plus_dm_smooth / atr_1w, 0)
    minus_di_1w = np.where(atr_1w != 0, 100 * minus_dm_smooth / atr_1w, 0)
    
    dx_1w = np.where((plus_di_1w + minus_di_1w) != 0, 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 0)
    
    # Wilder's smoothing for ADX
    adx_1w = np.zeros_like(dx_1w)
    adx_1w[2*period_adx-1] = np.mean(dx_1w[period_adx-1:2*period_adx])
    for i in range(2*period_adx, len(dx_1w)):
        adx_1w[i] = (adx_1w[i-1] * (period_adx-1) + dx_1w[i]) / period_adx
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_12h[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_sma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Donchian high
        # 2. Strong weekly trend (ADX > 20)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i]) and \
           (adx_1w_aligned[i] > 20) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Donchian low
        # 2. Strong weekly trend (ADX > 20)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i]) and \
             (adx_1w_aligned[i] > 20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_1wADX20_Filter_v1"
timeframe = "12h"
leverage = 1.0