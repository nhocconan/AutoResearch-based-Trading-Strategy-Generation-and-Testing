#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot Breakout with weekly EMA200 trend filter and volume confirmation.
# Uses Camarilla levels (L3, H3) from prior day for breakout entries.
# Weekly EMA200 determines trend direction: long only when price > EMA200, short only when price < EMA200.
# Volume spike (>2x 20-period average) confirms breakout strength.
# Designed to capture momentum in trending markets while avoiding false breakouts in ranging conditions.
# Target: 15-25 trades/year to stay within optimal range for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(200) on weekly close
    ema_200_1w = np.full(len(df_1w), np.nan)
    alpha = 2 / (200 + 1)
    for i in range(len(close_1w)):
        if i < 199:
            ema_200_1w[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_200_1w[i-1]):
                ema_200_1w[i] = np.mean(close_1w[i-199:i+1])
            else:
                ema_200_1w[i] = close_1w[i] * alpha + ema_200_1w[i-1] * (1 - alpha)
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for calculations
    start_idx = 20  # for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous day's data
        if i >= 1:
            # Use previous day's OHLC (not current bar)
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            if range_val <= 0:
                signals[i] = 0.0
                continue
                
            camarilla_l3 = prev_close - (range_val * 1.1 / 6)
            camarilla_h3 = prev_close + (range_val * 1.1 / 6)
        else:
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_spike = False
        
        # Trend filter from weekly EMA200
        price_above_ema200 = close[i] > ema_200_1w_aligned[i]
        price_below_ema200 = close[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above H3 + above weekly EMA200 + volume spike
            if (close[i] > camarilla_h3 and 
                price_above_ema200 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below L3 + below weekly EMA200 + volume spike
            elif (close[i] < camarilla_l3 and 
                  price_below_ema200 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below L3 or trend turns down
            if (close[i] < camarilla_l3 or 
                not price_above_ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 or trend turns up
            if (close[i] > camarilla_h3 or 
                not price_below_ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_1wEMA200_Volume_v1"
timeframe = "1d"
leverage = 1.0