#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA200 Trend + Volume Spike
# Long when Bull Power > 0, price > 1d EMA200, and 1d volume > 1.5x 20-day average
# Short when Bear Power < 0, price < 1d EMA200, and 1d volume > 1.5x 20-day average
# Exit when Bull/Bear Power crosses zero or price crosses 1d EMA200
# Elder Ray measures bull/bear strength via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Works in bull (strong Bull Power) and bear (strong Bear Power) regimes
# Volume confirms conviction, EMA200 filters trend direction
# Target: 20-35 trades/year by requiring volume spike + trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13  # Bull Power: High - EMA13
    bear_power = low_1d - ema13   # Bear Power: Low - EMA13
    
    # Calculate 1d EMA200 for trend filter
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6-day high/low for exit (optional stop)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        price = close[i]
        ema200_val = ema200_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        volume = df_1d['volume'].iloc[i // 4] if i >= 4 else df_1d['volume'].iloc[0]  # 6 bars per day (24h/6h)
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma if i >= 4 else df_1d['volume'].iloc[0] > 1.5 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0, price > EMA200, volume confirmation
            if bull > 0 and price > ema200_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, price < EMA200, volume confirmation
            elif bear < 0 and price < ema200_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Bull Power <= 0 or price crosses below EMA200
                if bull <= 0 or price < ema200_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Bear Power >= 0 or price crosses above EMA200
                if bear >= 0 or price > ema200_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0