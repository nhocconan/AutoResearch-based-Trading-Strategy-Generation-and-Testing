#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Breakout_v1
Hypothesis: On 6h timeframe, combine Elder Ray (Bull/Bear Power) with 12h trend regime and volume confirmation. 
Elder Ray > 0 indicates bullish power, < 0 bearish power. Only take trades in direction of 12h EMA50 trend.
Volume must be above 20-period average to confirm breakout strength. Designed for low frequency (15-25/year) 
to work in both bull and bear markets via trend filter and power validation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for trend regime ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Elder Ray on 6h (primary) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 13-period EMA for Elder Ray (standard)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # === Volume filter ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) + above 12h EMA50 trend + volume confirmation
            if bull_val > 0 and price_close > ema_50_12h_val and vol_val > vol_ma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: Bear Power < 0 (bearish momentum) + below 12h EMA50 trend + volume confirmation
            elif bear_val < 0 and price_close < ema_50_12h_val and vol_val > vol_ma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit when power reverses or volume drops significantly
            if position == 1:
                # Exit long if Bull Power turns negative or volume drops below average
                if bull_val <= 0 or vol_val < vol_ma_val * 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if Bear Power turns positive or volume drops below average
                if bear_val >= 0 or vol_val < vol_ma_val * 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_Breakout_v1"
timeframe = "6h"
leverage = 1.0