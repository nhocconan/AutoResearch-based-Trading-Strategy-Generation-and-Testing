#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout at R1/S1 with volume confirmation and 12h EMA trend filter.
# Uses 12h EMA for trend direction to avoid counter-trend trades. Volume > 1.5x average confirms breakout strength.
# Camarilla levels provide precise entry/exit points. Target: 25-40 trades/year per symbol.
# Position size: 0.25 to manage risk during drawdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA (34-period) for trend
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation using 12h volume
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (4h close and 12h volume)
        price_close = prices['close'].iloc[i]
        vol_12h_current = align_htf_to_ltf(prices, df_12h, vol_12h)[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + volume surge + price > 12h EMA (uptrend)
            if (price_close > R1_aligned[i] and
                vol_12h_current > 1.5 * vol_ma_20_12h_aligned[i] and
                price_close > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + volume surge + price < 12h EMA (downtrend)
            elif (price_close < S1_aligned[i] and
                  vol_12h_current > 1.5 * vol_ma_20_12h_aligned[i] and
                  price_close < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price < S1 or trend turns down
                if (price_close < S1_aligned[i]) or (price_close < ema_34_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price > R1 or trend turns up
                if (price_close > R1_aligned[i]) or (price_close > ema_34_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0