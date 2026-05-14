#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_ATRRegime
Hypothesis: 4h strategy using Camarilla R1/S1 breakouts with 12h EMA50 trend filter, volume spike confirmation, and ATR-based regime filter (avoid low volatility). Designed for BTC/ETH robustness: trend alignment filters false breakouts, volume confirms participation, ATR regime avoids chop. Targets 80-120 trades over 4 years (20-30/year) with 0.25 position size. Uses discrete levels to minimize fee drag. Works in bull/bear via trend filter and volatility regime.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
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
    
    # Volume confirmation: current volume > 3.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (3.0 * vol_avg)
    
    # ATR-based regime filter: avoid extremely low volatility (choppy sideways)
    # ATR(20) > 20-period moving average of ATR(20) * 0.8
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr1])  # same length as close
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_regime = atr > (0.8 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 12h EMA50 (50), 1d shift(1) for Camarilla, vol avg (20), ATR (20+20 for EMA+MA)
    start_idx = max(50 + 1, 1 + 1, 20, 20 + 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(volatility_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        vol_reg = volatility_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 12h EMA50 alignment, volume confirmation, and volatility regime
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
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_ATRRegime"
timeframe = "4h"
leverage = 1.0