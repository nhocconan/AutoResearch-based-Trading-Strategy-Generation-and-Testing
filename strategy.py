#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_EMA20_Trend_V3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # === 1d: EMA20 for trend context ===
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # === 6h: Elder Ray components ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for power calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Volume filter: current vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        ema20_val = ema20_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ratio_val = vol_ratio[i]
        ema13_val = ema13[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_val) or np.isnan(bull_val) or np.isnan(bear_val) or 
            np.isnan(vol_ratio_val) or np.isnan(ema13_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + positive bull power + volume confirmation
            if (close_val > ema20_val and          # Price above 1d EMA20 (uptrend)
                bull_val > 0 and                   # Positive bull power
                vol_ratio_val > 1.3):              # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + negative bear power + volume confirmation
            elif (close_val < ema20_val and        # Price below 1d EMA20 (downtrend)
                  bear_val < 0 and                 # Negative bear power
                  vol_ratio_val > 1.3):            # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or power divergence
            if (close_val < ema20_val or           # Price below 1d EMA20
                bull_val < 0 or                    # Bull power turned negative
                vol_ratio_val < 0.7):              # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or power divergence
            if (close_val > ema20_val or           # Price above 1d EMA20
                bear_val > 0 or                    # Bear power turned positive
                vol_ratio_val < 0.7):              # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals