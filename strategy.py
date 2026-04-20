#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_EMA20_Trend_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Elder Ray Components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray (standard period)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 6h: EMA20 Trend Filter ===
    close_6h = prices['close'].values
    ema20_6h = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 6h: Volume Confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close_6h[i]
        ema_val = ema20_6h[i]
        bull_val = bull_power_1d_aligned[i]
        bear_val = bear_power_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power + price above EMA20 + volume confirmation
            if (bull_val > 0 and               # Positive bull power (bulls in control)
                close_val > ema_val and        # Price above EMA20 (uptrend)
                vol_ratio_val > 1.3):          # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power + price below EMA20 + volume confirmation
            elif (bear_val < 0 and             # Negative bear power (bears in control)
                  close_val < ema_val and      # Price below EMA20 (downtrend)
                  vol_ratio_val > 1.3):        # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bear power turns positive OR price breaks EMA20
            if (bear_val > 0 or                # Bears taking over
                close_val < ema_val):          # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bull power turns negative OR price breaks EMA20
            if (bull_val < 0 or                # Bulls taking over
                close_val > ema_val):          # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals