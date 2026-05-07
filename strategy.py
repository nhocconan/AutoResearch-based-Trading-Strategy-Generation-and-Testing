#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator (3 SMAs) with 1d trend filter.
# Long when ADX > 25 (trending), price above Alligator jaws (13-SMA shifted 8 forward),
# teeth (8-SMA shifted 5 forward) > lips (5-SMA shifted 3 forward), and 1d EMA50 rising.
# Short when ADX > 25, price below jaws, lips > teeth > jaws, and 1d EMA50 falling.
# Exit when ADX < 20 (no trend) or price crosses Alligator lines in opposite direction.
# Alligator identifies trend direction and entry; ADX filters for strong trends only.
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Designed for 6h timeframe with ~20-50 trades per year to minimize fee drag.
# Works in both bull and bear markets by only taking trades when strong trend exists (ADX>25)
# and aligned with 1d trend.

name = "6h_ADX_Alligator_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1]) if np.nansum(tr[1:period+1]) > 0 else 0
        plus_di[period] = np.nansum(plus_dm[1:period+1]) if np.nansum(plus_dm[1:period+1]) > 0 else 0
        minus_di[period] = np.nansum(minus_dm[1:period+1]) if np.nansum(minus_dm[1:period+1]) > 0 else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        dx = np.zeros_like(high)
        di_sum = plus_di + minus_di
        dx[di_sum > 0] = (np.abs(plus_di[di_sum > 0] - minus_di[di_sum > 0]) / di_sum[di_sum > 0]) * 100
        
        # ADX: smoothed DX
        adx = np.zeros_like(high)
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period if np.nansum(dx[period+1:2*period+1]) > 0 else 0
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Williams Alligator: 3 SMAs with future shifts
    # Jaws: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    def sma(series, period):
        return pd.Series(series).rolling(window=period, min_periods=period).mean().values
    
    jaws_raw = sma(close, 13)
    teeth_raw = sma(close, 8)
    lips_raw = sma(close, 5)
    
    # Shift forward (Alligator's predictive nature)
    jaws = np.full_like(jaws_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaws) > 8:
        jaws[8:] = jaws_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 13+8, 8+5, 5+3, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX > 25 (strong trend), price > jaws, teeth > lips > jaws (bullish alignment), 1d EMA50 rising
            long_cond = (adx[i] > 25) and (close[i] > jaws[i]) and (teeth[i] > lips[i]) and (lips[i] > jaws[i]) and ema50_rising[i]
            # Short conditions: ADX > 25, price < jaws, lips < teeth < jaws (bearish alignment), 1d EMA50 falling
            short_cond = (adx[i] > 25) and (close[i] < jaws[i]) and (lips[i] < teeth[i]) and (teeth[i] < jaws[i]) and ema50_falling[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 (weak trend) OR price crosses below jaws (trend change)
            if adx[i] < 20 or close[i] < jaws[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (weak trend) OR price crosses above jaws (trend change)
            if adx[i] < 20 or close[i] > jaws[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals