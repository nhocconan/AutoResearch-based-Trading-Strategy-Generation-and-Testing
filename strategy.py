#!/usr/bin/env python3
"""
exp_6675_6h_elder_ray_1w_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-week trend filter (EMA50) and volume confirmation.
In weekly uptrend (price > EMA50), go long when Bear Power < 0 and rising (bulls gaining control).
In weekly downtrend (price < EMA50), go short when Bull Power > 0 and falling (bears gaining control).
Uses 6h for entry timing, 1w for regime filter. Designed to work in both bull and bear markets
by aligning with the higher timeframe trend. Expects ~15-30 trades/year.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6675_6h_elder_ray_1w_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_PERIOD = 50          # 1-week EMA for trend filter
RAY_PERIOD = 13          # Elder Ray period (EMA for power calculation)
VOL_MA_PERIOD = 20       # Volume moving average
VOLUME_THRESHOLD = 1.5   # Volume must be 1.5x MA for confirmation
SIGNAL_SIZE = 0.25       # Position size (25% of capital)
ATR_PERIOD = 14          # ATR for stoploss
ATR_STOP_MULTIPLIER = 2.5 # Stoploss multiplier
MAX_HOLD_BARS = 8        # Maximum hold: ~4 days (8 * 6h = 48h)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA
    ema_ray = pd.Series(close).ewm(span=RAY_PERIOD, adjust=False, min_periods=RAY_PERIOD).mean().values
    bull_power = high - ema_ray
    bear_power = low - ema_ray
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    prev_bull_power = 0.0
    prev_bear_power = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, RAY_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine weekly trend regime
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOLUME_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Elder Ray signals with momentum (looking for improving power)
        bull_power_rising = bull_power[i] > prev_bull_power
        bear_power_falling = bear_power[i] < prev_bear_power
        
        # Long signal: weekly uptrend + bear power negative and rising (bulls gaining)
        long_signal = weekly_uptrend and (bear_power[i] < 0) and bull_power_rising and vol_confirmed
        
        # Short signal: weekly downtrend + bull power positive and falling (bears gaining)
        short_signal = weekly_downtrend and (bull_power[i] > 0) and bear_power_falling and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
            
        # Store previous power values for next iteration
        prev_bull_power = bull_power[i]
        prev_bear_power = bear_power[i]
    
    return signals