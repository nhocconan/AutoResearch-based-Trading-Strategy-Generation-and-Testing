#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS_v2
Hypothesis: Refined version with stricter volume confirmation (2.0x volume MA) and 
additional momentum filter (RSI > 50 for longs, < 50 for shorts) to reduce trades
while maintaining edge. Focuses on high-probability breakouts in trending markets.
Target: 20-30 trades/year to minimize fee drag on 4h timeframe.
"""

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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R1, S1 levels
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (available after daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 2.0 * 20-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # RSI filter for momentum confirmation
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, volume MA, and RSI
    start_idx = max(35, 20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        r1_level = R1_aligned[i]
        s1_level = S1_aligned[i]
        vol_confirm_val = vol_confirm[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: break above R1 with volume, uptrend, and bullish momentum
            if (close[i] > r1_level and vol_confirm_val and 
                close[i] > ema_trend and rsi_val > 50):
                signals[i] = size
                position = 1
            # Short: break below S1 with volume, downtrend, and bearish momentum
            elif (close[i] < s1_level and vol_confirm_val and 
                  close[i] < ema_trend and rsi_val < 50):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: close below S1 or trend turns down or momentum fades
            if (close[i] < s1_level or close[i] < ema_trend or rsi_val < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above R1 or trend turns up or momentum fades
            if (close[i] > r1_level or close[i] > ema_trend or rsi_val > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS_v2"
timeframe = "4h"
leverage = 1.0