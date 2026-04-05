#!/usr/bin/env python3
"""
Experiment #8479: 6h Williams Alligator + Elder Ray + ADX regime filter
Hypothesis: Combines trend (Alligator), momentum (Elder Ray), and regime (ADX) to capture strong trends while avoiding chop.
Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future shift.
Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
ADX > 25 indicates trending market. Only trade in trending regime.
Targets 75-200 total trades over 4 years (19-50/year) to balance frequency and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8479_6h_alligator_elder_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Blue line
ALLIGATOR_TEETH_PERIOD = 8  # Red line
ALLIGATOR_LIPS_PERIOD = 5   # Green line
ELDER_EMA_PERIOD = 13       # For Elder Ray calculation
ADX_PERIOD = 14
ADX_THRESHOLD = 25          # Trending regime threshold
SIGNAL_SIZE = 0.25

def calculate_ema(series, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for regime context)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter (optional context)
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, 50)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: SMAs with forward shift
    jaw = pd.Series(close).rolling(window=ALLIGATOR_JAW_PERIOD, min_periods=ALLIGATOR_JAW_PERIOD).mean().values
    teeth = pd.Series(close).rolling(window=ALLIGATOR_TEETH_PERIOD, min_periods=ALLIGATOR_TEETH_PERIOD).mean().values
    lips = pd.Series(close).rolling(window=ALLIGATOR_LIPS_PERIOD, min_periods=ALLIGATOR_LIPS_PERIOD).mean().values
    
    # Shift forward by periods/2 to avoid look-ahead (Alligator specification)
    jaw_shift = ALLIGATOR_JAW_PERIOD // 2
    teeth_shift = ALLIGATOR_TEETH_PERIOD // 2
    lips_shift = ALLIGATOR_LIPS_PERIOD // 2
    
    jaw = np.roll(jaw, -jaw_shift)
    teeth = np.roll(teeth, -teeth_shift)
    lips = np.roll(lips, -lips_shift)
    
    # Invalidate shifted values
    jaw[:jaw_shift] = np.nan
    teeth[:teeth_shift] = np.nan
    lips[:lips_shift] = np.nan
    
    # Elder Ray: Bull Power and Bear Power using EMA13
    ema13 = calculate_ema(close, ELDER_EMA_PERIOD)
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ADX for regime detection
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD, 
                ELDER_EMA_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Alligator not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > threshold (trending market)
        if adx[i] < ADX_THRESHOLD:
            # In ranging market, stay flat or reduce position
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1])
        strong_bear = bear_power[i] > 0 and bear_power[i] > np.mean(bear_power[max(0, i-20):i+1])
        
        # Entry conditions
        long_entry = bullish_alignment and strong_bull
        short_entry = bearish_alignment and strong_bear
        
        # Exit conditions: Alligator reversal or weak Elder Ray
        long_exit = not bullish_alignment or (bull_power[i] <= 0)
        short_exit = not bearish_alignment or (bear_power[i] <= 0)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:  # Short
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals