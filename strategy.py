#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d ADX regime filter and volume spike
# Long when price breaks above Camarilla R1 (1d) + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S1 (1d) + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla levels provide precise intraday support/resistance that work in ranging markets.
# ADX > 25 filters for trending conditions only, reducing whipsaws in choppy markets.
# Volume threshold (1.5x) targets ~15-25 trades/year on 12h timeframe to avoid overtrading.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) and ADX ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (based on previous day)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # ADX calculation (14-period)
    # +DM = max(H - H_prev, 0) if H - H_prev > L_prev - L else 0
    # -DM = max(L_prev - L, 0) if L_prev - L > H - H_prev else 0
    # TR = max(H - L, abs(H - C_prev), abs(L - C_prev))
    # +DI = 100 * EWMA(+DM, 14) / EWMA(TR, 14)
    # -DI = 100 * EWMA(-DM, 14) / EWMA(TR, 14)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWMA(DX, 14)
    
    # Calculate +DM, -DM, TR
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    close_shift = np.roll(close_1d, 1)
    
    high_shift[0] = high_1d[0]
    low_shift[0] = low_1d[0]
    close_shift[0] = close_1d[0]
    
    plus_dm = np.where((high_1d - high_shift) > (low_shift - low_1d), np.maximum(high_1d - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low_1d) > (high_1d - high_shift), np.maximum(low_shift - low_1d, 0), 0)
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift), np.abs(low_1d - close_shift)))
    
    # EWMA smoothing (alpha = 1/14)
    alpha = 1.0 / 14.0
    tr_ewma = np.zeros_like(tr)
    plus_dm_ewma = np.zeros_like(plus_dm)
    minus_dm_ewma = np.zeros_like(minus_dm)
    
    tr_ewma[0] = tr[0]
    plus_dm_ewma[0] = plus_dm[0]
    minus_dm_ewma[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        tr_ewma[i] = alpha * tr[i] + (1 - alpha) * tr_ewma[i-1]
        plus_dm_ewma[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_ewma[i-1]
        minus_dm_ewma[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_ewma[i-1]
    
    # Avoid division by zero
    plus_di = np.where(tr_ewma != 0, 100 * plus_dm_ewma / tr_ewma, 0)
    minus_di = np.where(tr_ewma != 0, 100 * minus_dm_ewma / tr_ewma, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX = EWMA of DX
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align 1d indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h Indicators: Volume SMA ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # 1d lookback + Donchian equivalent + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Regime filter: ADX > 25 (trending market)
        trending = adx_aligned[i] > 25.0
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (close > R1)
        # 2. Trending market (ADX > 25)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1_aligned[i]) and \
           trending and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (close < S1)
        # 2. Trending market (ADX > 25)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1_aligned[i]) and \
             trending and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dADX_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0