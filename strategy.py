#!/usr/bin/env python3
"""
Experiment #12595: 6h Elder Ray + Weekly Regime Filter
Hypothesis: Elder Ray (Bull Power/Bear Power) identifies institutional buying/selling pressure.
Weekly regime filter (above/below 200 EMA) ensures we trade with the higher timeframe trend.
Works in bull markets via strong bull power + uptrend, and in bear via bear power + downtrend.
Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12595_6h_elder_ray_weekly_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_LENGTH = 13          # EMA length for Elder Ray
WEEKLY_EMA_LENGTH = 200        # Weekly trend filter
RSI_LENGTH = 14                # Entry timing filter
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SIGNAL_SIZE = 0.25             # Position size (25%)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    ema_weekly = calculate_ema(df_weekly['close'].values, WEEKLY_EMA_LENGTH)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components
    ema_close = calculate_ema(close, ELDER_RAY_LENGTH)
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    # RSI for entry timing
    rsi = calculate_rsi(close, RSI_LENGTH)
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup period
    start = max(ELDER_RAY_LENGTH, WEEKLY_EMA_LENGTH, RSI_LENGTH, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Determine weekly regime
        weekly_uptrend = close[i] > ema_weekly_aligned[i]
        weekly_downtrend = close[i] < ema_weekly_aligned[i]
        
        # Elder Ray signals with RSI filter
        strong_bull_power = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        strong_bear_power = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        rsi_not_overbought = rsi[i] < RSI_OVERBOUGHT
        rsi_not_oversold = rsi[i] > RSI_OVERSOLD
        
        # Entry conditions
        long_entry = (weekly_uptrend and strong_bull_power and rsi[i] < 50 and rsi_not_oversold)
        short_entry = (weekly_downtrend and strong_bear_power and rsi[i] > 50 and rsi_not_overbought)
        
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