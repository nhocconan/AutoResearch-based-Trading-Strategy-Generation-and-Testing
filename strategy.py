#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 breakout with 1w ADX trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R1 + 1w ADX > 20 + volume > 1.8x 20-period avg
# Short when price breaks below 12h Camarilla S1 + 1w ADX > 20 + volume > 1.8x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla pivots provide mathematically derived support/resistance. ADX filter ensures we only trade strong trends.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring ADX > 20.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
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
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === 12h Indicator: Camarilla Pivot Levels (based on previous 12h bar) ===
    # Calculate pivots using previous bar's OHLC
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate today's pivots
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        # Pivot point
        pivot = (high_prev + low_prev + close_prev) / 3.0
        
        # Camarilla levels
        range_prev = high_prev - low_prev
        camarilla_r1[i] = pivot + (range_prev * 1.1 / 12)
        camarilla_s1[i] = pivot - (range_prev * 1.1 / 12)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 2*period) + 20  # pivots need 1 bar, ADX(28) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Camarilla R1
        # 2. Trend (1w ADX > 20)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1[i]) and \
           (adx_aligned[i] > 20) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Camarilla S1
        # 2. Trend (1w ADX > 20)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1[i]) and \
             (adx_aligned[i] > 20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1wADX20_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0