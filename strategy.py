#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long: Bull Power > 0 AND Bear Power rising (less negative) AND 1d ADX > 25 (strong trend)
# - Short: Bear Power < 0 AND Bull Power falling (less positive) AND 1d ADX > 25 (strong trend)
# - Exit: Elder Ray signal reverses OR price crosses EMA(13) in opposite direction
# - Uses 1d ADX for regime filtering to avoid whipsaws in ranging markets
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by filtering only strong trends (ADX > 25)

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Elder Ray and ADX (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA(13)
    bear_power = low_1d - ema_13_1d   # Bear Power = Low - EMA(13)
    
    # Pre-compute 1d ADX for regime filter
    # True Range
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h EMA(13) for exit signal
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13_6h[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Elder Ray values
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Regime filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Elder Ray momentum: rising Bull Power or falling Bear Power
        # Need previous values to check momentum
        if i > 100:
            bull_power_prev = bull_power_aligned[i-1]
            bear_power_prev = bear_power_aligned[i-1]
            bull_power_rising = bull_power_val > bull_power_prev
            bear_power_falling = bear_power_val < bear_power_prev
        else:
            bull_power_rising = False
            bear_power_falling = False
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power positive AND rising AND strong trend
        if bull_power_val > 0 and bull_power_rising and strong_trend:
            enter_long = True
        
        # Short: Bear Power negative AND falling AND strong trend
        if bear_power_val < 0 and bear_power_falling and strong_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power becomes positive OR price crosses below EMA(13)
            exit_long = bear_power_val > 0 or close_price < ema_13_6h[i]
        elif position == -1:
            # Exit short if Bull Power becomes negative OR price crosses above EMA(13)
            exit_short = bull_power_val < 0 or close_price > ema_13_6h[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals