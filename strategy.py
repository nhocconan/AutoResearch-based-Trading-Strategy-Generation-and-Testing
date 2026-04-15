#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h ADX regime filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + 12h ADX > 25 (strong trend)
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + 12h ADX > 25 (strong trend)
# Uses discrete position sizing (0.30) to balance return and drawdown. Designed for low trade frequency (20-40/year).
# Donchian channels provide objective structure; ADX filter ensures we only trade strong trends, avoiding chop and sideways markets.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring ADX > 25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # === 12h Indicator: ADX (regime filter) ===
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
    
    atr_12h = np.zeros_like(tr)
    if len(tr) >= period:
        atr_12h[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_12h[i] = (atr_12h[i-1] * (period-1) + tr[i]) / period
    
    plus_di_12h = np.zeros_like(plus_dm)
    minus_di_12h = np.zeros_like(minus_dm)
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    if len(plus_dm) >= period:
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        for i in range(period, len(plus_dm)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Avoid division by zero
    plus_di_12h = np.where(atr_12h != 0, 100 * plus_dm_smooth / atr_12h, 0)
    minus_di_12h = np.where(atr_12h != 0, 100 * minus_dm_smooth / atr_12h, 0)
    
    dx_12h = np.where((plus_di_12h + minus_di_12h) != 0, 
                      100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 0)
    
    # Wilder's smoothing for ADX
    adx_12h = np.zeros_like(dx_12h)
    if len(dx_12h) >= 2*period:
        adx_12h[2*period-1] = np.mean(dx_12h[period-1:2*period])
        for i in range(2*period, len(dx_12h)):
            adx_12h[i] = (adx_12h[i-1] * (period-1) + dx_12h[i]) / period
    
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 4h Indicators: Donchian(20) and Volume SMA ===
    # Donchian high/low (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 2*period)  # 28 for ADX, 20 for Donchian
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. Strong trend (12h ADX > 25)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           (adx_12h_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. Strong trend (12h ADX > 25)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             (adx_12h_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_ADX25_Filter_v2"
timeframe = "4h"
leverage = 1.0