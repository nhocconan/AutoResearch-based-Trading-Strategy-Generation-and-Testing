#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX regime filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period volume SMA + 1d ADX > 20 (trending market)
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period volume SMA + 1d ADX > 20 (trending market)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Donchian channels provide objective breakout levels. ADX filter ensures we only trade in trending regimes, avoiding chop.
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation) by requiring ADX > 20.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ADX (trend strength filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    plus_dm = np.where((high_1d - high_1d_shift) > (low_1d_shift - low_1d), 
                       np.maximum(high_1d - high_1d_shift, 0), 0)
    minus_dm = np.where((low_1d_shift - low_1d) > (high_1d - high_1d_shift), 
                        np.maximum(low_1d_shift - low_1d, 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
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
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h Indicators: Donchian Channel and Volume SMA ===
    donchian_period = 20
    vol_sma_period = 20
    
    # Donchian high/low
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=vol_sma_period, min_periods=vol_sma_period).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_period, vol_sma_period, 2*period) + 10
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_sma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Donchian high
        # 2. Trending market (ADX > 20)
        # 3. Volume confirmation
        if (close[i] > highest_high[i]) and \
           (adx_aligned[i] > 20) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Donchian low
        # 2. Trending market (ADX > 20)
        # 3. Volume confirmation
        elif (close[i] < lowest_low[i]) and \
             (adx_aligned[i] > 20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_ADX20_Filter_v1"
timeframe = "12h"
leverage = 1.0