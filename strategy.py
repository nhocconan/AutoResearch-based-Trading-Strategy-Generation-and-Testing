#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 12h trend filter (EMA50).
# Camarilla levels from 1d provide high-probability support/resistance levels.
# 12h EMA(50) trend filter ensures trades align with intermediate-term direction.
# Volume > 1.3x average confirms institutional participation at breakout.
# Works in bull/bear as 12h EMA adapts to trend and Camarilla adapts to volatility.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend filter
    ema_len = 50
    if len(df_12h) < ema_len:
        return np.zeros(n)
    
    ema_12h = pd.Series(df_12h['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Camarilla levels from 1d (previous day)
    # Calculate using previous day's high, low, close
    phigh = df_1d['high'].shift(1).values  # previous day high
    plow = df_1d['low'].shift(1).values    # previous day low
    pclose = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla formulas
    range_ = phigh - plow
    camarilla_h4 = pclose + range_ * 1.1/2
    camarilla_l4 = pclose - range_ * 1.1/2
    camarilla_h3 = pclose + range_ * 1.1/4
    camarilla_l3 = pclose - range_ * 1.1/4
    camarilla_h2 = pclose + range_ * 1.1/6
    camarilla_l2 = pclose - range_ * 1.1/6
    camarilla_h1 = pclose + range_ * 1.1/12
    camarilla_l1 = pclose - range_ * 1.1/12
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # need 12h EMA50 and volume MA20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or
            np.isnan(l2_aligned[i]) or
            np.isnan(h1_aligned[i]) or
            np.isnan(l1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla H3/H4 + above 12h EMA + volume
            if ((close[i] > h3_aligned[i] or close[i] > h4_aligned[i]) and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below Camarilla L3/L4 + below 12h EMA + volume
            elif ((close[i] < l3_aligned[i] or close[i] < l4_aligned[i]) and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Camarilla L3 or breaks below L4
            if close[i] < l3_aligned[i] or close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Camarilla H3 or breaks above H4
            if close[i] > h3_aligned[i] or close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_EMA50_1d_Camarilla_Volume_v1"
timeframe = "4h"
leverage = 1.0