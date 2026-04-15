#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(12) zero-line crossover with 1d volume spike filter and ADX regime filter
# Long when TRIX crosses above zero + 1d volume > 1.5x 20-period avg + 1d ADX > 25 (trending)
# Short when TRIX crosses below zero + 1d volume > 1.5x 20-period avg + 1d ADX > 25 (trending)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# TRIX is a momentum oscillator that filters noise and captures sustained moves.
# Volume confirmation ensures breakouts have participation.
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Target: ~20-30 trades/year on 12h timeframe to avoid overtrading.

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
    
    # === 1d Indicators: EMA for TRIX, Volume SMA, ADX ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # EMA smoothing for TRIX (typical periods: 12)
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix_1d = trix_raw.fillna(0).values
    
    # Volume SMA for confirmation
    vol_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    # +DM, -DM, TR
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = pd.Series(high_1d) - pd.Series(low_1d)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm_smooth / atr_14
    minus_di = 100 * minus_dm_smooth / atr_14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean().fillna(0).values
    
    # Align 1d indicators to 12h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 14) + 5  # TRIX needs 3*12 + buffer, ADX needs 14*2 + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period volume SMA (use aligned)
        vol_confirm = vol_1d[i] > (vol_sma_20_1d_aligned[i] * 1.5) if i < len(vol_1d) else False
        
        # ADX filter: trending market (ADX > 25)
        trending = adx_1d_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. TRIX crosses above zero (trix[i] > 0 and trix[i-1] <= 0)
        # 2. Volume confirmation
        # 3. Trending market (ADX > 25)
        if i > 0:
            trix_now = trix_1d_aligned[i]
            trix_prev = trix_1d_aligned[i-1]
            if (trix_now > 0) and (trix_prev <= 0) and vol_confirm and trending:
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. TRIX crosses below zero (trix[i] < 0 and trix[i-1] >= 0)
        # 2. Volume confirmation
        # 3. Trending market (ADX > 25)
        elif i > 0:
            trix_now = trix_1d_aligned[i]
            trix_prev = trix_1d_aligned[i-1]
            if (trix_now < 0) and (trix_prev >= 0) and vol_confirm and trending:
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_TRIX12_1dVolume_ADX_Filter_v1"
timeframe = "12h"
leverage = 1.0