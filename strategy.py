#!/usr/bin/env python3
name = "6h_Chaikin_Money_Flow_Regime_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Chaikin Money Flow (CMF) on 6h data
    # CMF = (Sum of Money Flow Volume over N periods) / (Sum of Volume over N periods)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    
    # Avoid division by zero
    high_low = high - low
    high_low[high_low == 0] = 1e-10
    
    mfm = ((close - low) - (high - close)) / high_low
    mfv = mfm * volume
    
    # 20-period CMF
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum
    cmf[vol_sum == 0] = 0
    
    # Calculate 6-period RSI for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    
    rs = avg_loss.copy()
    rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(cmf[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CMF > 0.1 (strong buying pressure) + above weekly EMA20 + RSI < 70 (not overbought)
            if cmf[i] > 0.1 and close[i] > ema_20_1w_aligned[i] and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.1 (strong selling pressure) + below weekly EMA20 + RSI > 30 (not oversold)
            elif cmf[i] < -0.1 and close[i] < ema_20_1w_aligned[i] and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CMF turns negative or breaks below weekly EMA20
            if cmf[i] < 0 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CMF turns positive or breaks above weekly EMA20
            if cmf[i] > 0 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals