#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Confluence_v1
Hypothesis: Elder Ray (Bull/Bear Power) combined with 12h trend regime and volume confirmation captures sustained moves while filtering chop. Works in bull/bear by only taking longs in bull regime (EMA50>EMA200) and shorts in bear regime (EMA50<EMA200). Low trade frequency (~20-40/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMAs for regime filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # === Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === Volume confirmation (20-period on 6h) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_spike = vol_ratio[i]
        ema_50 = ema_50_12h_aligned[i]
        ema_200 = ema_200_12h_aligned[i]
        
        # Regime: bull if EMA50 > EMA200, bear if EMA50 < EMA200
        bull_regime = ema_50 > ema_200
        bear_regime = ema_50 < ema_200
        
        if position == 0:
            # Long: bull regime + bull power > 0 + volume spike > 1.8
            if bull_regime and bull > 0 and vol_spike > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: bear regime + bear power < 0 + volume spike > 1.8
            elif bear_regime and bear < 0 and vol_spike > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: regime change or Elder Power divergence
            if position == 1:
                # Exit long: regime turns bear OR bull power turns negative
                if not bull_regime or bull <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: regime turns bull OR bear power turns positive
                if not bear_regime or bear >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_Confluence_v1"
timeframe = "6h"
leverage = 1.0