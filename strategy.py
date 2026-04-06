# 1. STATEMENT OF HYPOTHESIS
# ============================================================
# Strategy: 4-hour Donchian(20) breakout with 12-hour EMA trend filter and volume confirmation.
# Rationale:
#   - Donchian breakouts capture momentum in trending markets (bull or bear).
#   - 12-hour EMA ensures trades align with the higher timeframe trend, reducing whipsaws.
#   - Volume confirmation filters out low-momentum breakouts.
#   - ATR-based stop-loss limits downside.
#   - Target: 75-200 total trades over 4 years (19-50/year) to balance signal quality and fee drag.
#   - Works in bull markets (catching uptrends) and bear markets (catching downtrends).
#   - Tested successfully on SOLUSDT in prior experiments; aims to generalize to BTC/ETH.
# ============================================================

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13253_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 20      # 12-hour EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25   # 25% of capital per trade
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EWM with alpha=1/period)."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Use pandas EWM for Wilder's smoothing: alpha = 1/period
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12-hour data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12-hour EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4-hour indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0          # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of all lookbacks)
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not yet available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Volume confirmation: current volume > threshold * average volume
        volume_ok = (not np.isnan(volume_ma[i])) and (volume[i] > volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Breakout signals: price breaks Donchian channel with volume and trend alignment
        # Note: Use previous bar's Donchian levels to avoid look-ahead
        breakout_up = volume_ok and uptrend and (high[i] > highest_high[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < lowest_low[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE  # Maintain long position
        elif position == -1:
            signals[i] = -SIGNAL_SIZE  # Maintain short position
    
    return signals