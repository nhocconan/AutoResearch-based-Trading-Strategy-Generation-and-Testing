#!/usr/bin/env python3
"""
Experiment #7934: 1-hour momentum with 4h trend filter and volume confirmation.
Hypothesis: 1h momentum breakouts in direction of 4h trend with volume >1.5x MA capture
sustained moves while avoiding counter-trend whipsaws. 4h trend filter reduces false
signals in ranging markets. Session filter (08-20 UTC) focuses on active hours.
Target: 60-150 total trades over 4 years (15-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7934_1h_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_LENGTH = 10
VOLUME_MA_LENGTH = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
TREND_EMA_LENGTH = 21
ATR_LENGTH = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=TREND_EMA_LENGTH, adjust=False, min_periods=TREND_EMA_LENGTH).mean().values
    
    # Trend: 1 = bullish (close > EMA), -1 = bearish (close < EMA)
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Momentum: close - close[n periods ago]
    momentum = close - np.roll(close, MOMENTUM_LENGTH)
    # Set first MOMENTUM_LENGTH values to 0 (no look-ahead)
    momentum[:MOMENTUM_LENGTH] = 0
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_LENGTH, min_periods=VOLUME_MA_LENGTH).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_LENGTH, adjust=False, min_periods=ATR_LENGTH).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_LENGTH, VOLUME_MA_LENGTH, ATR_LENGTH, TREND_EMA_LENGTH) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                # Check stop loss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]):
            if position != 0:
                # Check stop loss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
        if position == 1:
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = SIGNAL_SIZE
                continue
        elif position == -1:
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -SIGNAL_SIZE
                continue
        
        # Momentum and volume conditions
        mom_up = momentum[i] > 0
        mom_down = momentum[i] < 0
        vol_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: momentum in direction of 4h trend
        long_entry = mom_up and vol_ok and (trend_4h_aligned[i] == 1)
        short_entry = mom_down and vol_ok and (trend_4h_aligned[i] == -1)
        
        # Generate signals
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
    
    return signals