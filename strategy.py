#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_Filter_v1
Hypothesis: 6h Camarilla R4/S4 breakouts with 1w trend filter. In strong weekly trends (price > 1w EMA50), only take breakouts in trend direction (R4 longs in uptrend, S4 shorts in downtrend). In ranging weekly markets (price near 1w EMA50), take both breakout directions for mean reversion potential. Uses volume confirmation to avoid false breakouts. Targets 50-150 trades over 4 years by requiring weekly alignment and volume spike. Works in bull/bear via adaptive logic: trend continuation in strong trends, bidirectional in ranging weeks. Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, 
                        np.where(close < ema_50_1w_aligned, -1, 0))  # 1=uptrend, -1=downtrend, 0=ranging
    
    # Calculate ATR for volume spike filter
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 1.5 * ATR-adjusted average volume
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Calculate Camarilla levels from previous 6h bar
    camarilla_r4 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    
    for i in range(1, n):
        # Camarilla levels based on previous bar's range
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r4[i] = prev_close + (range_val * 1.1 / 2)  # R4 = C + (H-L)*1.1/2
        camarilla_s4[i] = prev_close - (range_val * 1.1 / 2)  # S4 = C - (H-L)*1.1/2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for camarilla)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation
        if not vol_spike[i]:
            # No volume spike - hold current position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout logic with weekly trend filter
        if htf_trend[i] == 1:  # Weekly uptrend
            if close[i] > camarilla_r4[i]:  # R4 breakout - continuation long
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] < camarilla_s4[i]:  # S4 breakdown - avoid shorts in uptrend
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
                    
        elif htf_trend[i] == -1:  # Weekly downtrend
            if close[i] < camarilla_s4[i]:  # S4 breakdown - continuation short
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            elif close[i] > camarilla_r4[i]:  # R4 breakout - avoid longs in downtrend
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
                    
        else:  # Weekly ranging (near EMA50)
            if close[i] > camarilla_r4[i]:  # R4 breakout - long
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] < camarilla_s4[i]:  # S4 breakdown - short
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0