#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 reversal with 1d EMA34 trend filter and volume confirmation.
# Long when price closes below S4 (oversold) AND price > 1d EMA34 with volume spike.
# Short when price closes above R4 (overbought) AND price < 1d EMA34 with volume spike.
# Uses mean-reversion at extreme Camarilla levels (R4/S4) with trend alignment to avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation on reversal. Designed for ~20-30 trades/year.
# Works in both bull and bear markets by fading extremes in the direction of 1d trend.
name = "4h_Camarilla_R4S4_1dEMA34_Reversal"
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
    
    # Load 1d data ONCE for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (H, L, C of completed day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 4h volume average for spike detection (20-period EMA)
    vol_ema_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long condition: close below S4 (oversold), in uptrend with volume spike
            long_condition = (close[i] < S4_aligned[i]) and uptrend and vol_spike[i]
            # Short condition: close above R4 (overbought), in downtrend with volume spike
            short_condition = (close[i] > R4_aligned[i]) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses above S4 (mean reversion complete) or trend turns down
            if (close[i] > S4_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses below R4 (mean reversion complete) or trend turns up
            if (close[i] < R4_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals