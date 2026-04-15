#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(12) zero-cross with 1d EMA50 trend filter and volume spike
# Long when TRIX crosses above zero + 1d EMA50 uptrend + volume > 1.6x 20-period avg
# Short when TRIX crosses below zero + 1d EMA50 downtrend + volume > 1.6x 20-period avg
# TRIX is a triple-smoothed EMA momentum oscillator effective in ranging markets
# Combined with EMA50 trend filter to avoid counter-trend trades
# Volume confirmation ensures participation
# Target: 15-25 trades/year on 12h timeframe to minimize fee drag

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
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Indicator: TRIX(12) ===
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate TRIX as percentage change of triple EMA
    trix = np.zeros_like(close)
    trix[12:] = (ema3[12:] - ema3[11:-1]) / ema3[11:-1] * 100  # Avoid division by zero
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 36) + 5  # EMA50 + TRIX(12) needs 36 periods + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or  # Need previous for crossover
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.6x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.6)
        
        # TRIX zero-cross detection
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        # === LONG CONDITIONS ===
        # 1. TRIX crosses above zero
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if trix_cross_up and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. TRIX crosses below zero
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif trix_cross_down and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_TRIX12_1dEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0