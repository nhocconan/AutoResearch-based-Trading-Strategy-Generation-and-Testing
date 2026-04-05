#!/usr/bin/env python3
"""
exp_7559_6d_triple_confluence_v1
Hypothesis: 6-hour trend following with triple confluence: 12h Supertrend for direction, 
6h momentum (RSI) for entry timing, and volume confirmation. Only takes trades when 
all three align, reducing false signals. Uses ATR-based trailing stops. Designed for 
low frequency (target: 50-150 total trades over 4 years) to minimize fee impact.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7559_6d_triple_confluence_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
TRAIL_ATR_MULTIPLIER = 2.5

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    upper_band = upper_band.copy()
    lower_band = lower_band.copy()
    
    for i in range(1, len(close)):
        if upper_band[i] > upper_band[i-1] or close[i-1] < upper_band[i-1]:
            upper_band[i] = upper_band[i-1]
        if lower_band[i] < lower_band[i-1] or close[i-1] > lower_band[i-1]:
            lower_band[i] = lower_band[i-1]
    
    supertrend = np.zeros(len(close))
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = upper_band[i]
        else:
            if supertrend[i-1] == upper_band[i-1]:
                supertrend[i] = lower_band[i] if close[i] <= upper_band[i] else upper_band[i]
            else:
                supertrend[i] = upper_band[i] if close[i] >= lower_band[i] else lower_band[i]
    
    direction = np.where(close > supertrend, 1, -1)
    return supertrend, direction, atr.values

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Supertrend for trend direction
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    supertrend_12h, dir_12h, atr_12h = calculate_supertrend(
        high_12h, low_12h, close_12h, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER
    )
    dir_12h_aligned = align_htf_to_ltf(prices, df_12h, dir_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI for momentum
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for trailing stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SUPERTREND_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(dir_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Update trailing stop for existing position
        if position == 1:  # long position
            new_stop = max(stop_price, high[i] - (TRAIL_ATR_MULTIPLIER * atr[i]))
            if close[i] <= new_stop:
                signals[i] = 0.0
                position = 0
                continue
            stop_price = new_stop
            signals[i] = SIGNAL_SIZE
        elif position == -1:  # short position
            new_stop = min(stop_price, low[i] + (TRAIL_ATR_MULTIPLIER * atr[i]))
            if close[i] >= new_stop:
                signals[i] = 0.0
                position = 0
                continue
            stop_price = new_stop
            signals[i] = -SIGNAL_SIZE
        else:
            signals[i] = 0.0
        
        # Skip if already in position (handled above)
        if position != 0:
            continue
            
        # Check confluence conditions
        trend_up = dir_12h_aligned[i] == 1
        trend_down = dir_12h_aligned[i] == -1
        
        # RSI conditions (avoid extremes, look for momentum)
        rsi_not_overbought = rsi[i] < RSI_OVERBOUGHT
        rsi_not_oversold = rsi[i] > RSI_OVERSOLD
        rsi_momentum_up = (i > 0 and rsi[i] > rsi[i-1] and rsi[i] < 60)
        rsi_momentum_down = (i > 0 and rsi[i] < rsi[i-1] and rsi[i] > 40)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = trend_up and rsi_momentum_up and volume_confirmed
        short_entry = trend_down and rsi_momentum_down and volume_confirmed
        
        # Generate signals
        if long_entry:
            signals[i] = SIGNAL_SIZE
            position = 1
            entry_price = close[i]
            stop_price = entry_price - (TRAIL_ATR_MULTIPLIER * atr[i])
        elif short_entry:
            signals[i] = -SIGNAL_SIZE
            position = -1
            entry_price = close[i]
            stop_price = entry_price + (TRAIL_ATR_MULTIPLIER * atr[i])
    
    return signals