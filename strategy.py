#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX Regime with 12h EMA50 trend filter
# Long when Bull Power > 0, ADX > 25 (trending), and close > 12h EMA50
# Short when Bear Power < 0, ADX > 25 (trending), and close < 12h EMA50
# Exit when ADX < 20 (range regime) or trend EMA crossover
# Uses discrete position sizing (0.25) to balance capture and risk.
# Elder Ray measures bull/bear power via EMA13, ADX filters for trending markets.
# 12h EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading in strong trends (ADX>25) with the 12h trend.

name = "6h_ElderRay_ADX_Regime_12hEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Bull/Bear Power (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 13, 50)  # ADX, Elder Power, and 12h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_adx = adx[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: ADX < 20 (range regime) OR trend EMA crossover (close < 12h EMA50)
            if curr_adx < 20.0 or curr_close < curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (range regime) OR trend EMA crossover (close > 12h EMA50)
            if curr_adx < 20.0 or curr_close > curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0, ADX > 25 (strong trend), and close > 12h EMA50
            if curr_bull > 0 and curr_adx > 25.0 and curr_close > curr_ema50_12h:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0, ADX > 25 (strong trend), and close < 12h EMA50
            elif curr_bear < 0 and curr_adx > 25.0 and curr_close < curr_ema50_12h:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals