#!/usr/bin/env python3
# 1d_WilliamsAlligator_ElderRay_Vortex_Volume
# Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend direction, Elder Ray measures bull/bear power,
# Vortex confirms trend strength, and volume spike filters for institutional participation.
# Works in bull markets via long when bull power > 0 and price above teeth; in bear markets via short when bear power > 0 and price below teeth.
# Uses 1d timeframe with 1w trend filter to reduce noise and false signals.

name = "1d_WilliamsAlligator_ElderRay_Vortex_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1w Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Williams Alligator on weekly: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    # Using EMA as proxy for SMMA with same period
    jaw_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_1w = pd.Series(close_1w).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    jaw_1d = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1d = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1d = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # === Elder Ray Power (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # === Vortex Indicator (14-period) ===
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], tr])
    
    vm_plus = np.abs(high - low[1:])
    vm_plus = np.concatenate([[vm_plus[0]], vm_plus])
    vm_minus = np.abs(low - high[1:])
    vm_minus = np.concatenate([[vm_minus[0]], vm_minus])
    
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = sum_vm_plus14 / sum_tr14
    vi_minus = sum_vm_minus14 / sum_tr14
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(lips_1d[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull power > 0, price above teeth, VI+ > VI-, and volume spike
            if bull_power[i] > 0 and close[i] > teeth_1d[i] and vi_plus[i] > vi_minus[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power > 0, price below teeth, VI- > VI+, and volume spike
            elif bear_power[i] > 0 and close[i] < teeth_1d[i] and vi_minus[i] > vi_plus[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull power <= 0 or price below lips or trend change (VI- > VI+)
            if bull_power[i] <= 0 or close[i] < lips_1d[i] or vi_minus[i] > vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear power <= 0 or price above lips or trend change (VI+ > VI-)
            if bear_power[i] <= 0 or close[i] > lips_1d[i] or vi_plus[i] > vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals