#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Keltner Channel + Momentum + Volume Confirmation
# Uses 4h as primary timeframe with 1d trend filter (EMA34) and volume spike (>1.5x average)
# Long when: price > upper Keltner channel, positive momentum, above 1d EMA34, volume confirmed
# Short when: price < lower Keltner channel, negative momentum, below 1d EMA34, volume confirmed
# Keltner Channel: EMA20 ± (ATR10 * 2)
# Momentum: ROC10 (Rate of Change over 10 periods)
# Volume confirmation ensures institutional participation in breakouts
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years)

name = "4h_Keltner_Momentum_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA20 for Keltner middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR10 for Keltner width
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Keltner Channel bands
    upper_keltner = ema20 + (2 * atr10)
    lower_keltner = ema20 - (2 * atr10)
    
    # Calculate ROC10 momentum
    roc10 = np.zeros(n)
    for i in range(10, n):
        roc10[i] = ((close[i] - close[i-10]) / close[i-10]) * 100
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Need EMA20 and ROC10 data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or np.isnan(atr10[i]) or 
            np.isnan(roc10[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        momentum = roc10[i]
        ema_trend = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price > upper Keltner, positive momentum, above 1d EMA34
            if price > upper and momentum > 0 and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price < lower Keltner, negative momentum, below 1d EMA34
            elif price < lower and momentum < 0 and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price < EMA20 OR momentum <= 0 OR below 1d EMA34
            if price < ema20[i] or momentum <= 0 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price > EMA20 OR momentum >= 0 OR above 1d EMA34
            if price > ema20[i] or momentum >= 0 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals