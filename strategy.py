#!/usr/bin/env python3
"""
Experiment #7679: 6-hour Ichimoku Cloud with 12-hour Kumo Twist Filter and Volume Confirmation.
Hypothesis: In bull markets (Tenkan > Kijun and price above Kumo), go long on TK cross up.
In bear markets (Tenkan < Kijun and price below Kumo), go short on TK cross down.
Volume must be above 1.5x average to confirm momentum.
Uses 6h for entry timing, 12h for trend filter, targeting 50-150 total trades (12-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7679_6h_ichimoku_12h_kumo_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9
KJ_PERIOD = 26
SENB_SPAN_B_PERIOD = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5  # volume must be 1.5x average
SIGNAL_SIZE = 0.25

def generate_signals(prices):
    n = len(prices)
    if n < SENB_SPAN_B_PERIOD:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Ichimoku components
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_12h).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                  pd.Series(low_12h).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_12h).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                 pd.Series(low_12h).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(KJ_PERIOD)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_12h).rolling(window=SENB_SPAN_B_PERIOD, min_periods=SENB_SPAN_B_PERIOD).max() + 
                      pd.Series(low_12h).rolling(window=SENB_SPAN_B_PERIOD, min_periods=SENB_SPAN_B_PERIOD).min()) / 2).shift(KJ_PERIOD)
    
    # Align HTF Ichimoku to LTF
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b.values)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # LTF Tenkan-sen and Kijun-sen for TK cross
    tenkan_sen_ltf = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                      pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    kijun_sen_ltf = (pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                     pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    
    # TK cross signals (current vs previous)
    tk_cross_up = (tenkan_sen_ltf > kijun_sen_ltf) & (tenkan_sen_ltf.shift(1) <= kijun_sen_ltf.shift(1))
    tk_cross_down = (tenkan_sen_ltf < kijun_sen_ltf) & (tenkan_sen_ltf.shift(1) >= kijun_sen_ltf.shift(1))
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SENB_SPAN_B_PERIOD, VOLUME_MA_PERIOD) + KJ_PERIOD + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Determine market regime using 12h Ichimoku
        bull_regime = (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) and \
                      (close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i]))
        bear_regime = (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) and \
                      (close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i]))
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_regime and tk_cross_up.iloc[i] if hasattr(tk_cross_up, 'iloc') else bool(tk_cross_up[i]) and volume_confirmed
        short_entry = bear_regime and tk_cross_down.iloc[i] if hasattr(tk_cross_down, 'iloc') else bool(tk_cross_down[i]) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals