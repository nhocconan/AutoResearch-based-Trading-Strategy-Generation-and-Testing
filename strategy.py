#!/usr/bin/env python3
"""
exp_6694_1h_trend_follow_4h_ema_v1
Hypothesis: 1h timeframe strategy using 4h EMA for trend direction and 1h EMA crossovers for entry timing.
Uses 4h EMA21 as primary trend filter (bullish when price > EMA21, bearish when price < EMA21).
Entries occur on 1h when EMA9 crosses above/below EMA21 in the direction of the 4h trend.
Volume confirmation (1.5x 20-period volume MA) reduces false breakouts.
Session filter (08-20 UTC) avoids low-liquidity periods.
Fixed position size of 0.20 to control risk and minimize fee churn.
Designed to capture medium-term trends while minimizing trades (target: 60-150 over 4 years).
Works in both bull and bear markets by following the 4h trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6694_1h_trend_follow_4h_ema_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
FAST_EMA = 9
SLOW_EMA = 21
VOL_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 4h for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=SLOW_EMA, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Fast and slow EMA on 1h
    ema_fast = pd.Series(close).ewm(span=FAST_EMA, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=SLOW_EMA, adjust=False).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], use DatetimeIndex for .hour
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(SLOW_EMA, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE  # hold position outside session
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine 4h trend direction
        # Bullish trend: price above 4h EMA21
        # Bearish trend: price below 4h EMA21
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]
        
        # EMA crossover signals on 1h
        # Golden cross: fast EMA crosses above slow EMA
        # Death cross: fast EMA crosses below slow EMA
        golden_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        death_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOLUME_THRESHOLD
        
        # Enter new positions only if flat
        if position == 0:
            if bullish_trend and golden_cross and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bearish_trend and death_cross and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals