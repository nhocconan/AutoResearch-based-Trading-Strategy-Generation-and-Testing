#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13). Long when Bull Power > 0 and rising, price above 1d EMA200, volume above average.
# Short when Bear Power < 0 and falling, price below 1d EMA200, volume above average.
# Uses ATR-based stop loss. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Elder Ray captures bull/bear power, EMA200 filters trend, volume confirms strength.

name = "exp_13819_6h_elderray12h_ema1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_EMA_PERIOD = 13
EMA_TREND_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Elder Ray calculations ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for Elder Ray
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, ELDER_EMA_PERIOD)
    
    # Align 12h EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Load 1d data for EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_TREND_PERIOD)
    
    # Align 1d EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data for Elder Ray components, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high - ema_12h_aligned
    bear_power = low - ema_12h_aligned
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_EMA_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Elder Ray signals with momentum
        # Long: Bull Power > 0 and rising (current > previous), price above 1d EMA, volume confirmation
        long_signal = (bull_power[i] > 0 and 
                      i > start and bull_power[i] > bull_power[i-1] and
                      above_ema and volume_ok)
        
        # Short: Bear Power < 0 and falling (current < previous), price below 1d EMA, volume confirmation
        short_signal = (bear_power[i] < 0 and 
                       i > start and bear_power[i] < bear_power[i-1] and
                       below_ema and volume_ok)
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when Bull Power turns negative
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when Bear Power turns positive
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals