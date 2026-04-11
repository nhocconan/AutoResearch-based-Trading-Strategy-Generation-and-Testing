#!/usr/bin/env python3
# 12h_1w_camarilla_breakout_v1
# Strategy: 12h Camarilla breakout with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Weekly trend provides strong directional bias. At 12h timeframe, price reacts to weekly Camarilla H3/L3 levels with volume confirmation. This captures institutional interest at key weekly support/resistance while avoiding counter-trend trades. Works in both bull/bear by trading breakouts in weekly trend direction, avoiding false signals in chop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Camarilla levels: H3/L3 = C +- (H-L)*1.1/2
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        # Long: Price breaks above H3 + above 1w EMA20 (uptrend) + volume confirmation
        if vol_confirmed and close[i] > camarilla_h3_aligned[i] and close[i] > ema_20_1w_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below L3 + below 1w EMA20 (downtrend) + volume confirmation
        elif vol_confirmed and close[i] < camarilla_l3_aligned[i] and close[i] < ema_20_1w_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to midpoint or trend reversal
        elif position == 1 and (close[i] < (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2 or close[i] < ema_20_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2 or close[i] > ema_20_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals