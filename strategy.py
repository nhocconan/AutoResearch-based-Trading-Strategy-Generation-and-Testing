# 6h_ElderRay_1dTrend_SwingReversal
# Elder Ray (Bull/Bear Power) with 1d EMA trend filter and swing reversal signals
# Works in bull (trend continuation) and bear (mean reversion at swings)
# Target: 15-35 trades/year per symbol, low turnover to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for trend and swing levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate 13-period EMA for Elder Ray (1d) ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # === Calculate swing high/low from 1d (2-bar lookback for confirmation) ===
    # Swing High: high > previous high AND high > next high
    # Swing Low: low < previous low AND low < next low
    swing_high_1d = np.zeros_like(high_1d, dtype=bool)
    swing_low_1d = np.zeros_like(low_1d, dtype=bool)
    
    for i in range(1, len(high_1d)-1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            swing_high_1d[i] = True
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            swing_low_1d[i] = True
    
    # === Align all 1d indicators to 6h timeframe ===
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    swing_high_1d_aligned = align_htf_to_ltf(prices, df_1d, swing_high_1d.astype(float))
    swing_low_1d_aligned = align_htf_to_ltf(prices, df_1d, swing_low_1d.astype(float))
    
    # === 6s EMA20 for entry confirmation ===
    ema20_6h = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(swing_high_1d_aligned[i]) or
            np.isnan(swing_low_1d_aligned[i]) or np.isnan(ema20_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema13 = ema13_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        swing_high = swing_high_1d_aligned[i] > 0.5
        swing_low = swing_low_1d_aligned[i] > 0.5
        ema20 = ema20_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit on bearish swing OR bear power turning negative
            if swing_low or bear_power < 0:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit on bullish swing OR bull power turning positive
            if swing_high or bull_power > 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull power positive AND price above EMA20 AND at swing low (reversal)
            if bull_power > 0 and price > ema20 and swing_low:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Bear power negative AND price below EMA20 AND at swing high (reversal)
            elif bear_power < 0 and price < ema20 and swing_high:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dTrend_SwingReversal"
timeframe = "6h"
leverage = 1.0