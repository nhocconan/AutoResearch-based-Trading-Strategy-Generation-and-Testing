#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation. 
Camarilla pivot levels provide high-probability reversal/breakout levels from institutional order flow. 
1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws. 
Volume spike (ATR ratio > 1.5) confirms institutional participation. 
Discrete sizing 0.25 limits trades to target 12-37/year. Works in bull/bear via 1d trend filter.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    
    # Calculate Camarilla levels from previous day
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    # Using previous 1d bar's OHLC
    camarilla_R1 = close_1d + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    camarilla_S1 = close_1d - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and ATR ratio, 1d data needs at least 1 bar)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_ratio[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike filter
        camarilla_R1_val = camarilla_R1_aligned[i]
        camarilla_S1_val = camarilla_S1_aligned[i]
        size = fixed_size
        
        # Entry conditions: Camarilla R1/S1 breakout with volume spike AND aligned with 1d EMA50 trend
        # Long: price breaks above R1 (bullish breakout)
        # Short: price breaks below S1 (bearish breakout)
        long_entry = (close_val > camarilla_R1_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < camarilla_S1_val) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters Camarilla range (H3-L3) or trend reversal
            # Calculate H3 and L3 for exit: H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
            camarilla_H3 = close_1d + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
            camarilla_L3 = close_1d - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
            camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
            camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
            
            h3_val = camarilla_H3_aligned[i]
            l3_val = camarilla_L3_aligned[i]
            
            if (close_val < camarilla_R1_val and close_val > camarilla_S1_val) or \
               (close_val < h3_val and close_val > l3_val) or \
               (close_val < ema_50_val):  # back inside range or trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Camarilla range or trend reversal
            camarilla_H3 = close_1d + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
            camarilla_L3 = close_1d - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
            camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
            camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
            
            h3_val = camarilla_H3_aligned[i]
            l3_val = camarilla_L3_aligned[i]
            
            if (close_val > camarilla_S1_val and close_val < camarilla_R1_val) or \
               (close_val > l3_val and close_val < h3_val) or \
               (close_val > ema_50_val):  # back inside range or trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0