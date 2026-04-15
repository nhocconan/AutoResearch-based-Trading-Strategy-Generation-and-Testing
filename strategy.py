#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action at 1d VWAP with volume spike and 1w EMA200 trend filter
# Long when price > 1d VWAP + volume > 2x 20-period avg + 1w EMA200 uptrend (price > EMA200)
# Short when price < 1d VWAP + volume > 2x 20-period avg + 1w EMA200 downtrend (price < EMA200)
# Uses discrete position sizing (0.25) to minimize fee drag. VWAP acts as dynamic support/resistance.
# 1w EMA200 provides strong long-term trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~20-30 trades/year to minimize fee drag.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: VWAP ===
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume'].values).cumsum() / df_1d['volume'].values.cumsum()
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === 1w Indicator: EMA200 ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(200, 20) + 5  # EMA200 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price above 1d VWAP
        # 2. 1w EMA200 uptrend (price > EMA200)
        # 3. Volume confirmation
        if (close[i] > vwap_1d_aligned[i]) and \
           (close[i] > ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price below 1d VWAP
        # 2. 1w EMA200 downtrend (price < EMA200)
        # 3. Volume confirmation
        elif (close[i] < vwap_1d_aligned[i]) and \
             (close[i] < ema_200_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_VWAP_1wEMA200_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0