#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = -(HH - Close) / (HH - LL) * 100
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    williams_r = np.zeros_like(close_1d)
    hh_minus_ll = highest_high - lowest_low
    williams_r = np.where(hh_minus_ll != 0, -(highest_high - close_1d) / hh_minus_ll * 100, -50)
    
    # Calculate 1d Williams %R EMA (9-period)
    williams_ema = pd.Series(williams_r).ewm(span=9, adjust=False).values
    
    # Calculate 6h price position relative to 1d VWAP approximation
    # Approximate VWAP using typical price * volume / cumulative volume
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = typical_price * df_1d['volume'].values
    vwap_denominator = np.cumsum(df_1d['volume'].values)
    vwap = np.where(vwap_denominator != 0, np.cumsum(vwap_numerator) / vwap_denominator, typical_price)
    
    # Calculate 6h returns for momentum
    close_6h = prices['close'].values
    returns = np.zeros_like(close_6h)
    returns[1:] = (close_6h[1:] - close_6h[:-1]) / close_6h[:-1]
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    williams_ema_aligned = align_htf_to_ltf(prices, df_1d, williams_ema)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(williams_ema_aligned[i]) or
            np.isnan(vwap_aligned[i]) or np.isnan(returns[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        current_price = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: Williams %R oversold AND price below VWAP AND positive momentum
            if (williams_r_aligned[i] < -80 and
                williams_r_aligned[i] > williams_ema_aligned[i] and  # Williams crossing above EMA
                current_price < vwap_aligned[i] and
                returns[i] > 0):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought AND price above VWAP AND negative momentum
            elif (williams_r_aligned[i] > -20 and
                  williams_r_aligned[i] < williams_ema_aligned[i] and  # Williams crossing below EMA
                  current_price > vwap_aligned[i] and
                  returns[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R crosses EMA in opposite direction OR price crosses VWAP
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams crosses below EMA OR price crosses above VWAP
                if (williams_r_aligned[i] < williams_ema_aligned[i] or
                    current_price > vwap_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams crosses above EMA OR price crosses below VWAP
                if (williams_r_aligned[i] > williams_ema_aligned[i] or
                    current_price < vwap_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_VWAP_Cross"
timeframe = "6h"
leverage = 1.0