#!/usr/bin/env python3
# 1d_ema_cross_1w_volume_v1
# Hypothesis: 1d strategy using EMA crossover (21/55) for trend, volume confirmation (>1.5x 20-day avg), and 1w HTF trend filter (price > 1w EMA50). 
# Long: EMA21 > EMA55 + volume confirmed + close > 1w EMA50
# Short: EMA21 < EMA55 + volume confirmed + close < 1w EMA50
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-100 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_cross_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # EMA21 and EMA55 on 1d
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA crossover down OR price below 1w EMA50
            if ema_21[i] < ema_55[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA crossover up OR price above 1w EMA50
            if ema_21[i] > ema_55[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: EMA21 > EMA55 + price above 1w EMA50
                if ema_21[i] > ema_55[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: EMA21 < EMA55 + price below 1w EMA50
                elif ema_21[i] < ema_55[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals