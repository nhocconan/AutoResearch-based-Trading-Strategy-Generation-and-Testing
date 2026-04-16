#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Regime: ADX(14) > 25 = Trending, ADX < 20 = Range
# Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend)
# Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong downtrend)
# Exit when ADX < 20 (regime change to ranging) or power signals weaken
# Uses 1d ADX for regime to avoid whipsaws, 6h Elder Ray for entry timing
# Target: 60-180 total trades over 4 years (15-45/year) with discrete size 0.25
# Works in bull markets via trend following, avoids bear markets via regime filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Indicators: Elder Ray Components ===
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = ema13 - low   # Bear Power = EMA - Low
    
    # === 1d Indicators: ADX for Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bull = bull_power[i]
        bear = bear_power[i]
        adx_val = adx_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if regime changes to ranging (ADX < 20) or bull power weakens
            if adx_val < 20.0 or bull <= 0.0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if regime changes to ranging (ADX < 20) or bear power weakens
            if adx_val < 20.0 or bear <= 0.0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Strong uptrend - bull power positive, bear power negative, trending regime
            if bull > 0.0 and bear < 0.0 and adx_val > 25.0:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Strong downtrend - bear power positive, bull power negative, trending regime
            elif bear > 0.0 and bull < 0.0 and adx_val > 25.0:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dADX_RegimeFilter_V1"
timeframe = "6h"
leverage = 1.0