#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# Camarilla levels provide statistically significant intraday support/resistance.
# CHOP filter ensures we only trade in ranging markets where mean reversion works.
# Volume spike confirms institutional interest at pivot levels.
# Works in ranging markets (2025 BTC/ETH bear/range) by fading extreme moves to proven levels.

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
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 4h Indicators: Choppiness Index and Volume SMA ===
    # Choppiness Index (14-period)
    chop_window = 14
    atr_14 = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    # ATR calculation with min_periods
    atr_14[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, n):
        atr_14[i] = (atr_14[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Sum of ATR over chop_window period
    sum_atr_14 = np.zeros(n)
    sum_atr_14[chop_window-1] = np.sum(atr_14[:chop_window])
    for i in range(chop_window, n):
        sum_atr_14[i] = sum_atr_14[i-1] - atr_14[i-chop_window] + atr_14[i]
    
    # Max-min range over chop_window period
    max_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    min_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    range_chop = max_high - min_low
    
    # Choppiness Index: 100 * log10(sum(ATR)/range) / log10(chop_window)
    chop = np.zeros(n)
    for i in range(chop_window-1, n):
        if range_chop[i] > 0 and sum_atr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / range_chop[i]) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral value
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(chop_window, 20) + 5  # CHOP(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Choppiness filter: CHOP > 61.8 (strong ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Chop filter (range market)
        if (close[i] > camarilla_r3_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Chop filter (range market)
        elif (close[i] < camarilla_s3_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_CamarillaR3S3_1dVol2x_CHOP_Filter_v1"
timeframe = "4h"
leverage = 1.0