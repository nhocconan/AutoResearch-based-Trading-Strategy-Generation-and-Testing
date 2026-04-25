#!/usr/bin/env python3
"""
6h Elder Ray Index with 1d ADX Regime Filter and Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
In strong trends (ADX > 25 on 1d), trade in direction of Elder Ray power. In ranging markets (ADX < 20),
fade extreme Elder Ray readings. Volume spike confirms conviction. Designed for 6h timeframe to target
12-37 trades/year (50-150 over 4 years) with discrete sizing to minimize fee drag.
"""

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
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX for regime detection (trending vs ranging)
    # Calculate +DI, -DI, DX
    period = 14
    up_move = df_1d['high'].diff()
    down_move = -df_1d['low'].diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    # Align ADX to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 and ADX calculation
    start_idx = max(13, 30)  # EMA13 + ADX warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        
        # Regime detection
        trending_market = adx_val > 25
        ranging_market = adx_val < 20
        
        if position == 0:
            # Look for entry signals
            if trending_market:
                # In trending markets: trade with Elder Ray power
                long_entry = (bull_power[i] > 0) and vol_spike
                short_entry = (bear_power[i] > 0) and vol_spike
            elif ranging_market:
                # In ranging markets: fade extreme Elder Ray readings
                # Long when bear power is extreme (oversold), short when bull power is extreme (overbought)
                long_entry = (bear_power[i] > np.percentile(bear_power[max(0, i-50):i+1], 80)) and vol_spike
                short_entry = (bull_power[i] > np.percentile(bull_power[max(0, i-50):i+1], 80)) and vol_spike
            else:
                # Transition regime: require stronger signals
                long_entry = (bull_power[i] > 0) and (bull_power[i] > np.percentile(bull_power[max(0, i-50):i+1], 70)) and vol_spike
                short_entry = (bear_power[i] > 0) and (bear_power[i] > np.percentile(bear_power[max(0, i-50):i+1], 70)) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Elder Ray turns negative OR loss of volume momentum
            if (bull_power[i] <= 0) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power turns negative OR loss of volume momentum
            if (bear_power[i] <= 0) or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0