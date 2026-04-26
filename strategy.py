#!/usr/bin/env python3
"""
6h_ADX_Alligator_ElderRay_Combo_v1
Hypothesis: On 6h timeframe, combine ADX trend strength, Williams Alligator crossover, and Elder Ray (Bull/Bear Power) for high-conviction entries. 
- Long when: ADX > 25 (strong trend), Alligator jaws < teeth < lips (bullish alignment), and Bull Power > 0 (bulls in control)
- Short when: ADX > 25, Alligator jaws > teeth > lips (bearish alignment), and Bear Power < 0 (bears in control)
- Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades in ranging markets.
- Discrete sizing (0.0, ±0.25) to limit fee churn. Target: 15-30 trades/year per symbol.
- Designed to work in both bull (trend following) and bear (avoiding false breakouts via ADX filter) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator (SMAs with specific periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA for simplicity (SMMA approximates SMA with proper warmup)
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().values  # simplified
    teeth = pd.Series(high).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(high).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13  # negative when bears in control
    
    # ADX calculation (14-period)
    # +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    up_move = np.concatenate([[0], up_move])
    down_move = np.concatenate([down_move, [0]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Handle division by zero
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need sufficient data for all indicators
    start_idx = max(50, 14, 13)  # EMA50, ADX, Elder Ray
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: 1d EMA50
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Alligator alignment
        alligator_bullish = jaw[i] < teeth[i] and teeth[i] < lips[i]
        alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray
        bulls_in_control = bull_power[i] > 0
        bears_in_control = bear_power[i] < 0
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: strong trend + bullish Alligator + bulls in control + 1d uptrend
            long_signal = strong_trend and alligator_bullish and bulls_in_control and trend_uptrend
            
            # Short: strong trend + bearish Alligator + bears in control + 1d downtrend
            short_signal = strong_trend and alligator_bearish and bears_in_control and trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend weakens OR Alligator reverses OR bears take control
            if (not strong_trend) or (not alligator_bullish) or (not bulls_in_control):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend weakens OR Alligator reverses OR bulls take control
            if (not strong_trend) or (not alligator_bearish) or (not bears_in_control):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_ElderRay_Combo_v1"
timeframe = "6h"
leverage = 1.0