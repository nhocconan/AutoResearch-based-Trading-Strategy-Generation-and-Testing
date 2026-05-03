#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Regime Filter. Uses Bull Power (high-EMA13) and Bear Power (low-EMA13)
# with ADX regime detection to avoid whipsaws. Long when Bull Power > 0 and ADX > 25 (trending up),
# Short when Bear Power < 0 and ADX > 25 (trending down). Flat when ADX < 20 (range) or opposing power.
# Designed for low trade frequency (12-37/year) with discrete sizing 0.25 to survive 2022 crash.
# Works in bull/bear via regime filter: only takes strong trend signals, avoids chop.

name = "6h_ElderRay_ADX_Regime_0.25"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate ADX(14) for regime filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime and entry conditions
        adx_val = adx[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Long: Bull Power > 0 AND ADX > 25 (strong uptrend)
        long_entry = (bull > 0) and (adx_val > 25)
        # Short: Bear Power < 0 AND ADX > 25 (strong downtrend)
        short_entry = (bear < 0) and (adx_val > 25)
        # Exit: ADX < 20 (range) OR opposing power (bull<0 for long, bear>0 for short)
        long_exit = (adx_val < 20) or (bull < 0)
        short_exit = (adx_val < 20) or (bear > 0)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals