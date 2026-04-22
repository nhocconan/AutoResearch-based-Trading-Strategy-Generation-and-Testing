#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action near 1d VWAP with 1w trend filter and 1d volume confirmation
# VWAP acts as dynamic support/resistance. Price near VWAP with volume confirmation
# indicates institutional interest. 1w EMA50 trend filter ensures alignment with
# higher timeframe momentum. Works in bull markets by capturing bounces from VWAP
# in uptrend and in bear markets by avoiding counter-trend trades. Targets 12-37 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Load 1d data for VWAP and volume (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d VWAP calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price near 1d VWAP (within 0.5%) + 1w uptrend + 1d volume spike
            vwap_dist = abs(close[i] - vwap_1d_aligned[i]) / vwap_1d_aligned[i]
            if (vwap_dist < 0.005 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price near 1d VWAP (within 0.5%) + 1w downtrend + 1d volume spike
            elif (vwap_dist < 0.005 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price moves away from VWAP (>1.0%) or trend reversal
            vwap_dist = abs(close[i] - vwap_1d_aligned[i]) / vwap_1d_aligned[i]
            if position == 1:
                if (vwap_dist > 0.01 or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (vwap_dist > 0.01 or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_VWAP_1wEMA50_1dVolSpike"
timeframe = "12h"
leverage = 1.0