#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with 1-day trend filter (ADX > 25) and volume confirmation.
# Long when: Close > H3 (Camarilla resistance) AND ADX > 25 AND volume > 1.5 * EMA20(volume).
# Short when: Close < L3 (Camarilla support) AND ADX > 25 AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the 50-period EMA or ADX drops below 20.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "12h_Camarilla_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels based on previous 12h bar (using current bar's high/low/close)
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    hl_range = high - low
    h3 = close + 1.1 * hl_range / 2
    l3 = close - 1.1 * hl_range / 2
    
    # EMA50 for exit
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(ema_50[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > H3 AND ADX > 25 AND volume spike
            long_condition = (close[i] > h3[i]) and (adx_aligned[i] > 25) and volume_spike[i]
            # Short: Close < L3 AND ADX > 25 AND volume spike
            short_condition = (close[i] < l3[i]) and (adx_aligned[i] > 25) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA50 or ADX drops below 20
            if close[i] < ema_50[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA50 or ADX drops below 20
            if close[i] > ema_50[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals