#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based breakout with 12h trend filter and volume confirmation.
# Long when price breaks above ATR-based upper band AND price > 12h EMA50 with volume spike.
# Short when price breaks below ATR-based lower band AND price < 12h EMA50 with volume spike.
# Uses 12h EMA50 trend filter to align with higher timeframe trend and avoid counter-trend trades.
# ATR-based bands adapt to volatility, providing dynamic breakout levels.
# Volume spike filter ensures momentum confirmation. Designed for fewer trades (target: 25-35/year) to reduce fee drag.
# Works in both bull and bear markets by following the 12h trend direction.
name = "6h_ATR_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h trend filter: 50-period EMA on close
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for dynamic breakout bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-based breakout bands (multiplier = 2.0)
    upper_band = close + 2.0 * atr
    lower_band = close - 2.0 * atr
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long condition: break above ATR upper band, in uptrend with volume spike
            long_condition = (close[i] > upper_band[i]) and uptrend and vol_spike[i]
            # Short condition: break below ATR lower band, in downtrend with volume spike
            short_condition = (close[i] < lower_band[i]) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below ATR lower band or trend turns down
            if (close[i] < lower_band[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above ATR upper band or trend turns up
            if (close[i] > upper_band[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals