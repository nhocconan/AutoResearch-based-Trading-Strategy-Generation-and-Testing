#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Bands Mean Reversion + 1w Trend Filter + Volume Spike
# In bear markets (2025+), price tends to revert to mean after extreme deviations.
# Bollinger Bands (20,2) identify overbought/oversold conditions.
# 1w EMA50 trend filter ensures we trade counter-trend only in ranging markets.
# Volume spike confirms exhaustion of the move.
# Target: 10-25 trades/year (40-100 over 4 years) to minimize fee drag.
name = "1d_BollingerMeanRev_1wTrendFilter_VolumeSpike"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(ema50_1w_1d[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        # Trend condition: only trade counter-trend when 1w trend is weak
        # In strong trends (price far from EMA50), avoid mean reversion
        trend_strength = abs(close[i] - ema50_1w_1d[i]) / ema50_1w_1d[i]
        weak_trend = trend_strength < 0.05  # Less than 5% deviation from 1w EMA50
        
        if position == 0:
            # Long: price below lower BB + weak trend + volume spike (exhausted selling)
            if close[i] < lower_bb[i] and weak_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price above upper BB + weak trend + volume spike (exhausted buying)
            elif close[i] > upper_bb[i] and weak_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above middle band OR trend strengthens
            if close[i] > sma20[i] or trend_strength >= 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below middle band OR trend strengthens
            if close[i] < sma20[i] or trend_strength >= 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals