# Python code
#!/usr/bin/env python3
"""
4h_Vortex_Trend_Volume
Hypothesis: Vortex indicator with volume confirmation and 1d trend filter captures
trending moves while filtering noise in both bull and bear markets.
Vortex VI+ > VI- indicates bullish trend, VI- > VI+ indicates bearish trend.
Entry on Vortex crossover with volume spike and 1d trend alignment.
Exit on opposite Vortex crossover.
Target: 20-50 trades/year per symbol.
"""

name = "4h_Vortex_Trend_Volume"
timeframe = "4h"
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
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex Indicator: VI+ and VI- over 14 periods
    vm_plus = np.abs(high[1:] - low[:-1])  # |HIGH - LOWprev|
    vm_minus = np.abs(low[1:] - high[:-1])  # |LOW - HIGHprev|
    
    # Sum over 14 periods
    n_period = 14
    sum_vm_plus = np.zeros(n)
    sum_vm_minus = np.zeros(n)
    sum_tr = np.zeros(n)
    
    for i in range(n_period, n):
        sum_vm_plus[i] = np.sum(vm_plus[i-n_period+1:i+1])
        sum_vm_minus[i] = np.sum(vm_minus[i-n_period+1:i+1])
        sum_tr[i] = np.sum(tr[i-n_period+1:i+1])
    
    # Handle initial period
    for i in range(1, n_period):
        sum_vm_plus[i] = np.sum(vm_plus[1:i+1]) if i >= 1 else 0.0
        sum_vm_minus[i] = np.sum(vm_minus[1:i+1]) if i >= 1 else 0.0
        sum_tr[i] = np.sum(tr[1:i+1]) if i >= 1 else 0.0
    
    vi_plus = np.where(sum_tr > 0, sum_vm_plus / sum_tr, 0.0)
    vi_minus = np.where(sum_tr > 0, sum_vm_minus / sum_tr, 0.0)
    
    # Vortex trend: VI+ > VI- = bullish, VI- > VI+ = bearish
    bullish_vortex = vi_plus > vi_minus
    bearish_vortex = vi_minus > vi_plus
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1d = df_1d['close'].values > ema_20_1d
    downtrend_1d = df_1d['close'].values < ema_20_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        bullish_v = bullish_vortex[i]
        bearish_v = bearish_vortex[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: VI+ crosses above VI- with 1d uptrend and volume confirmation
            if bullish_v and not bullish_vortex[i-1] and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ with 1d downtrend and volume confirmation
            elif bearish_v and not bearish_vortex[i-1] and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ (trend change to bearish)
            if bearish_v and not bearish_vortex[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- (trend change to bullish)
            if bullish_v and not bullish_vortex[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals