#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_1dTrend_WeeklyVolatility"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Get weekly data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Daily trend filter: EMA34 slope
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly volatility filter: ATR(14) normalized by price
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = high_1w[0] - close_1w[0]
    tr3[0] = low_1w[0] - close_1w[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_norm_1w = atr_14_1w / close_1w
    atr_norm_aligned = align_htf_to_ltf(prices, df_1w, atr_norm_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_norm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0 (both sides agree), 
            #       Uptrend (EMA34 rising), Low volatility environment
            long_cond = (bull_power_aligned[i] > 0 and 
                        bear_power_aligned[i] < 0 and
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        atr_norm_aligned[i] < 0.02)  # Below 2% weekly ATR
            
            # Short: Bear Power < 0, Bull Power < 0 (both negative),
            #        Downtrend (EMA34 falling), Low volatility environment
            short_cond = (bear_power_aligned[i] < 0 and 
                         bull_power_aligned[i] < 0 and
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         atr_norm_aligned[i] < 0.02)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power becomes positive (bulls losing control) OR volatility spikes
            if bear_power_aligned[i] > 0 or atr_norm_aligned[i] > 0.035:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power becomes positive (bears losing control) OR volatility spikes
            if bull_power_aligned[i] > 0 or atr_norm_aligned[i] > 0.035:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals