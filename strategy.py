#!/usr/bin/env python3
"""
Experiment #8919: 6h Elder Ray Index + 12h/1d Regime Filter + Volume Confirmation.
Hypothesis: Elder Ray (bull/bear power) captures institutional buying/selling pressure; 
combined with 12h/1d regime (trending vs ranging) and volume confirmation, 
this filters false signals and captures sustained moves in both bull and bear markets.
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8919_6h_elder_ray_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
REGIME_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(series, period):
    """Calculate EMA"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, REGIME_PERIOD)
    
    # Calculate 1d EMA for regime filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, REGIME_PERIOD)
    
    # Price relative to EMAs: above = bullish bias, below = bearish bias
    regime_12h = np.where(close_12h > ema_12h, 1, 
                   np.where(close_12h < ema_12h, -1, 0))
    regime_1d = np.where(close_1d > ema_1d, 1, 
                   np.where(close_1d < ema_1d, -1, 0))
    
    # Combine regimes: both must agree for strong signal
    strong_bull = (regime_12h == 1) & (regime_1d == 1)
    strong_bear = (regime_12h == -1) & (regime_1d == -1)
    
    # Align to 6t timeframe
    strong_bull_aligned = align_htf_to_ltf(prices, df_12h, strong_bull.astype(float))
    strong_bear_aligned = align_htf_to_ltf(prices, df_12h, strong_bear.astype(float))
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components
    ema_close = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, REGIME_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(strong_bull_aligned[i]) or np.isnan(strong_bear_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime from 12h/1d EMA
        bull_regime = strong_bull_aligned[i] == 1.0   # Both 12h and 1d bullish
        bear_regime = strong_bear_aligned[i] == 1.0   # Both 12h and 1d bearish
        
        # Elder Ray signals
        bull_pressure = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        bear_pressure = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_regime and bull_pressure and volume_confirmed
        short_entry = bear_regime and bear_pressure and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals