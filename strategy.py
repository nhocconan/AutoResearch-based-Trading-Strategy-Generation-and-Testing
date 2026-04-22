#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Commodity Channel Index (CCI) with 1-day EMA trend filter and volume confirmation.
# CCI identifies overbought (>100) and oversold (<-100) conditions with mean reversion tendencies.
# 1-day EMA provides trend direction: only take longs when price > 1d EMA, shorts when price < 1d EMA.
# Volume confirmation requires current volume > 1.3x 20-period average to filter weak signals.
# Designed to work in both bull and bear markets by aligning with trend via 1d EMA filter.
# Targets 20-40 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d data
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate CCI on 4h data (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    typical_price = (high + low + close) / 3.0
    ma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    deviation = np.abs(typical_price - ma_tp)
    mean_deviation = pd.Series(deviation).rolling(window=20, min_periods=20).mean().values
    
    # Avoid division by zero
    cci = np.where(mean_deviation != 0, (typical_price - ma_tp) / (0.015 * mean_deviation), 0.0)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(cci[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        cci_val = cci[i]
        ema_val = ema_1d_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long conditions: oversold + uptrend + volume spike
            if cci_val < -100 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought + downtrend + volume spike
            elif cci_val > 100 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when CCI returns to overbought or trend breaks
                if cci_val > 100 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when CCI returns to oversold or trend breaks
                if cci_val < -100 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_CCI_1dEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0