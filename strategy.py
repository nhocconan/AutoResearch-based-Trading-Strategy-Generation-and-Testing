#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1w trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: bull_power = high - EMA13, bear_power = low - EMA13
# Long when bull_power > 0 and bear_power < 0, 1w trend up, volume > 1.5x avg
# Short when bear_power < 0 and bull_power > 0, 1w trend down, volume > 1.5x avg
# Uses discrete position size 0.25 to limit churn. Designed for 6h to capture swings
# while filtering with weekly trend to avoid counter-trend trades in both bull/bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_ElderRay_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on weekly for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate EMA13 for Elder Ray on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull power: high minus EMA
    bear_power = low - ema_13   # Bear power: low minus EMA
    
    # Align weekly EMA200 to 6h
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20, 200)  # Ensure EMA13, vol MA, and weekly EMA200 are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) 
            or np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_13_val = ema_13[i]
        bull = bull_power[i]
        bear = bear_power[i]
        ema_200_1w_val = ema_200_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Weekly trend filter: trend up if price > EMA200, down if price < EMA200
        weekly_uptrend = price > ema_200_1w_val
        weekly_downtrend = price < ema_200_1w_val
        
        if position == 0:
            # Enter long when bull power positive, bear power negative (bulls in control),
            # weekly uptrend, and volume confirmation
            if bull > 0 and bear < 0 and weekly_uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short when bear power negative, bull power positive (bears in control),
            # weekly downtrend, and volume confirmation
            elif bear < 0 and bull > 0 and weekly_downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when bear power becomes positive (bears take over) or weekly trend turns down
            if bear > 0 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when bull power becomes positive (bulls take over) or weekly trend turns up
            if bull > 0 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals