#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Enhanced
# Hypothesis: On 4h timeframe, use daily Camarilla pivot levels (R1/S1) for breakout entries with 1d EMA34 trend filter and volume spike confirmation.
# Add ATR-based volatility filter to avoid choppy markets and increase robustness in both bull and bear markets.
# Enter long when price closes above R1 with volume > 2.0x 20-bar average, 1d EMA34 uptrend, and ATR ratio < 0.8 (low volatility).
# Enter short when price closes below S1 with volume > 2.0x 20-bar average, 1d EMA34 downtrend, and ATR ratio < 0.8.
# Exit when price crosses the 1d EMA34 (trend reversal) or ATR ratio > 1.2 (high volatility/chop).
# Targets 15-25 trades/year to minimize fee drag while capturing meaningful moves.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Enhanced"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot point and range
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla R1 and S1 levels
    r1 = daily_pivot + daily_range * 1.083
    s1 = daily_pivot - daily_range * 1.083
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: dynamic thresholds
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)  # Avoid division by zero
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma > 0, atr_ma, 1)  # Current ATR vs 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1d_trend = ema34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_ratio_val = atr_ratio[i]
        
        if position == 0:
            # LONG: Price closes above R1 with volume spike, 1d uptrend, and low volatility
            if (close[i] > r1_val and close[i] > ema1d_trend and 
                vol_ratio_val > 2.0 and atr_ratio_val < 0.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with volume spike, 1d downtrend, and low volatility
            elif (close[i] < s1_val and close[i] < ema1d_trend and 
                  vol_ratio_val > 2.0 and atr_ratio_val < 0.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d EMA34 (trend reversal) OR high volatility/chop
            if (close[i] < ema1d_trend) or (atr_ratio_val > 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d EMA34 (trend reversal) OR high volatility/chop
            if (close[i] > ema1d_trend) or (atr_ratio_val > 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals