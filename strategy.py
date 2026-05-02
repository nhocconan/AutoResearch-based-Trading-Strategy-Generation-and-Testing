#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX25 trend + volume confirmation
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# In strong trends (ADX > 25), take Elder Ray signals in trend direction
# Volume confirmation (1.5x 20-period average) filters weak breakouts
# Works in bull/bear by only taking signals aligned with 1d ADX trend
# Discrete sizing 0.25 targets 60-120 trades over 4 years (15-30/year)

name = "6h_ElderRay_1dADX25_Trend_Volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate ADX(25) on 1d
    plus_dm = np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0])
    minus_dm = np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0]))
    tr2 = np.abs(np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]))
    tr3 = np.abs(np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_period = 25
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate EMA13 on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation (1.5x 20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (vol_ma * 1.5)
        else:
            volume_spike = False
        
        if position == 0:  # Flat - look for new entries
            # Strong uptrend: ADX > 25 + Bull Power > 0
            long_signal = (adx_aligned[i] > 25) and (bull_power[i] > 0) and volume_spike
            # Strong downtrend: ADX > 25 + Bear Power > 0
            short_signal = (adx_aligned[i] > 25) and (bear_power[i] > 0) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 or ADX < 20 (trend weakening)
            if (bull_power[i] <= 0) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 or ADX < 20 (trend weakening)
            if (bear_power[i] <= 0) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals