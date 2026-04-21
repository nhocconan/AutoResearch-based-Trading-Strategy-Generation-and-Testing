#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R2/S2) breakout with 1w trend filter and volume confirmation.
# In strong trends (price > 1w EMA34), breakouts above R2 or below S2 have higher probability.
# Volume > 2.5x average confirms breakout strength. Target: 75-200 total trades over 4 years.
# Position size: 0.25 to manage risk during drawdowns. Works in bull/bear via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (R2, S2)
    high_prev = df_4h['high'].shift(1).values  # Previous 4h bar's high
    low_prev = df_4h['low'].shift(1).values    # Previous 4h bar's low
    close_prev = df_4h['close'].shift(1).values # Previous 4h bar's close
    
    # Camarilla equations for R2 and S2
    R2 = close_prev + 1.1 * (high_prev - low_prev) / 6
    S2 = close_prev - 1.1 * (high_prev - low_prev) / 6
    
    # Align Camarilla levels to current timeframe (no look-ahead)
    R2_aligned = align_htf_to_ltf(prices, df_4h, R2)
    S2_aligned = align_htf_to_ltf(prices, df_4h, S2)
    
    # Calculate 1-week EMA (34-period) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation using 4h volume
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_4h_current = align_htf_to_ltf(prices, df_4h, vol_4h)[i]
        
        if position == 0:
            # Enter long: price breaks above R2 + volume spike + price > 1w EMA (uptrend)
            if (price_close > R2_aligned[i] and
                vol_4h_current > 2.5 * vol_ma_20_4h_aligned[i] and
                price_close > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 + volume spike + price < 1w EMA (downtrend)
            elif (price_close < S2_aligned[i] and
                  vol_4h_current > 2.5 * vol_ma_20_4h_aligned[i] and
                  price_close < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S2 or trend turns down
                if (price_close < S2_aligned[i]) or (price_close < ema_34_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R2 or trend turns up
                if (price_close > R2_aligned[i]) or (price_close > ema_34_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R2_S2_Breakout_1wEMA34_Volume_Spike"
timeframe = "4h"
leverage = 1.0