#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and ADX > 25 trend filter
# Long when price breaks above 4h Donchian upper + 12h volume > 2.0x 20-period avg + 12h ADX > 25
# Short when price breaks below 4h Donchian lower + 12h volume > 2.0x 20-period avg + 12h ADX > 25
# Uses discrete position sizing (0.25) to minimize fee churn. Target 20-40 trades/year.
# Volume spike ensures momentum confirmation, ADX avoids chop, Donchian provides objective levels.
# Works in bull/bear markets by requiring strong trend (ADX>25) and volume confirmation.

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: ADX (trend strength filter) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_12h_shift = np.roll(high_12h, 1)
    low_12h_shift = np.roll(low_12h, 1)
    high_12h_shift[0] = high_12h[0]
    low_12h_shift[0] = low_12h[0]
    
    plus_dm = np.where((high_12h - high_12h_shift) > (low_12h_shift - low_12h), 
                       np.maximum(high_12h - high_12h_shift, 0), 0)
    minus_dm = np.where((low_12h_shift - low_12h) > (high_12h - high_12h_shift), 
                        np.maximum(low_12h_shift - low_12h, 0), 0)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Wilder's smoothing for ADX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 12h Indicator: Volume SMA (20-period) for spike detection ===
    vol_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 2*period) + 20  # Donchian(20) + ADX(28) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA (12h)
        vol_confirm = volume[i] > (vol_sma_20_12h_aligned[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Trend (12h ADX > 25)
        # 3. Volume spike confirmation
        if (close[i] > donchian_high[i]) and \
           (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Trend (12h ADX > 25)
        # 3. Volume spike confirmation
        elif (close[i] < donchian_low[i]) and \
             (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hADX25_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0