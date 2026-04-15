#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R1 + 1d EMA34 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S1 + 1d EMA34 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla levels from 1d provide institutional pivot points that work in ranging markets.
# 1d EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-30 trades/year on 4h timeframe to avoid overtrading.
# This combines proven Camarilla pivot breakout pattern with trend and volume filters.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) and EMA34 ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels calculation
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r1 = pivot + (range_1d * 1.1 / 12)
    camarilla_s1 = pivot - (range_1d * 1.1 / 12)
    
    # EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Indicators: Volume SMA ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # EMA34 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (close > R1)
        # 2. 1d EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1_aligned[i]) and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (close < S1)
        # 2. 1d EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1_aligned[i]) and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0