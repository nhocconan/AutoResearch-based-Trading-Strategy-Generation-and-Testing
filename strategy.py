# 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume
# Hypothesis: Camarilla pivot breakout with EMA trend filter and volume confirmation.
# Uses 1d Camarilla levels (R1/S1) on 4h chart, requiring price to close outside these levels.
# Trend filter: 1d EMA34 (price above for long, below for short).
# Volume filter: volume > 1.5x 30-period average.
# Risk management: exit on reverse breakout or volatility extremes (ATR ratio > 2.5 or < 0.5).
# Position size: 0.25 (25% of capital) to balance risk and reward.
# Designed for 4h timeframe to target 20-50 trades/year, avoiding overtrading.
# Should work in both bull and bear markets due to trend filter and volatility adaptation.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla levels, EMA, ATR, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla levels: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]  # handle first value
    range_1d = high_1d - low_1d
    R1 = close_1d_prev + range_1d * 1.1 / 12
    S1 = close_1d_prev - range_1d * 1.1 / 12
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: volume / 30-period average volume (1d)
    vol_ma_30 = pd.Series(df_1d['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_30
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        r1_level = R1_aligned[i]
        s1_level = S1_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be above average
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Enter long: price closes above R1, uptrend, volume spike, moderate volatility
            if (price_close > r1_level and 
                price_close > ema_trend and 
                vol_ratio > vol_threshold and 
                atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = 0.25
                position = 1
            # Enter short: price closes below S1, downtrend, volume spike, moderate volatility
            elif (price_close < s1_level and 
                  price_close < ema_trend and 
                  vol_ratio > vol_threshold and 
                  atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extremes
            if position == 1 and (price_close < s1_level or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > r1_level or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0