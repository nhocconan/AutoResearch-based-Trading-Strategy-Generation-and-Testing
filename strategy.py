#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_Regime
Hypothesis: 12h strategy using Camarilla R1/S1 breakouts from 1d pivot levels, filtered by 1w EMA50 trend direction and volume spike confirmation. Uses ATR-based volatility regime filter to avoid low-volatility whipsaws. Designed for BTC/ETH robustness: 1w trend filter ensures alignment with higher timeframe momentum, volume confirms breakout participation, volatility regime avoids chop. Targets 50-150 total trades over 4 years (12-37/year) with 0.25 position size. Uses discrete levels to minimize fee drag. Works in bull/bear via 1w trend filter and volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla R1/S1 levels (from previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.08)   # R1 level
    s1 = prev_close - (rng * 1.08)   # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.5 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    # ATR-based regime filter: avoid extremely low volatility (choppy sideways)
    # ATR(30) > 30-period moving average of ATR(30) * 0.7
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr1])  # same length as close
    atr = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ma = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
    volatility_regime = atr > (0.7 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 1w EMA50 (50), 1d shift(1) for Camarilla, vol avg (30), ATR (30+30 for EMA+MA)
    start_idx = max(50 + 1, 1 + 1, 30, 30 + 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(volatility_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        vol_reg = volatility_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1w EMA50 alignment, volume confirmation, and volatility regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            vol_reg)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             vol_reg)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_Regime"
timeframe = "12h"
leverage = 1.0