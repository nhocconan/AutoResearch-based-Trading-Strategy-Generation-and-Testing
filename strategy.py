#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA200 Trend Filter and Volume Spike
# Bull Power = High - EMA(200), Bear Power = EMA(200) - Low
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) and 1d volume > 1.5x 20-period average and price > 1d EMA200
# Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) and 1d volume > 1.5x 20-period average and price < 1d EMA200
# Exit when Bull Power and Bear Power have same sign (momentum divergence) or price crosses 1d EMA200
# Elder Ray measures bull/bear power relative to trend (EMA200), effective in both bull and bear markets
# Volume spike confirms conviction, EMA200 filter ensures trading with higher timeframe trend
# Target: 20-40 trades/year by requiring volume spike + clear Elder Ray signal + EMA200 filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(200)
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Elder Ray components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power = high_1d - ema200  # Bull Power = High - EMA200
    bear_power = ema200 - low_1d   # Bear Power = EMA200 - Low
    
    # Align to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if data not ready
        if np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (use 1d volume for 6f bar)
        price = close[i]
        # Approximate 1d volume for 6h bar (4 6h bars per day)
        idx_1d = i // 4
        if idx_1d >= len(df_1d):
            idx_1d = len(df_1d) - 1
        volume_1d = df_1d['volume'].iloc[idx_1d]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = volume_1d > 1.5 * vol_ma
        
        # Elder Ray signals
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Trend filter: price relative to EMA200
        price_above_ema200 = price > ema200_aligned[i]
        price_below_ema200 = price < ema200_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0 (bullish momentum), price > EMA200, volume spike
            if bull_power_val > 0 and bear_power_val < 0 and price_above_ema200 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0 (bearish momentum), price < EMA200, volume spike
            elif bear_power_val > 0 and bull_power_val < 0 and price_below_ema200 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions:
            # 1. Elder Ray divergence (both powers same sign = loss of momentum)
            # 2. Price crosses EMA200 (trend change)
            bull_power_val = bull_power_aligned[i]
            bear_power_val = bear_power_aligned[i]
            
            exit_signal = False
            
            # Condition 1: Loss of momentum (both powers same sign)
            if (bull_power_val > 0 and bear_power_val > 0) or (bull_power_val < 0 and bear_power_val < 0):
                exit_signal = True
            
            # Condition 2: Price crosses EMA200
            elif position == 1 and price_below_ema200:  # Long position, price below EMA200
                exit_signal = True
            elif position == -1 and price_above_ema200:  # Short position, price above EMA200
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