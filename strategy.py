#!/usr/bin/env python3
"""
exp_6678_1d_donchian20_1w_ema_vol_v1
Hypothesis: 1d Donchian(20) breakout with 1-week EMA trend filter and volume confirmation.
Uses weekly EMA(50) to determine trend direction (above = bullish, below = bearish).
Only takes breakouts in the direction of weekly trend to avoid counter-trend trades.
Volume confirmation ensures breakouts have conviction. ATR-based stoploss manages risk.
Designed for 1d timeframe to capture multi-day trends with minimal trades (~15-25/year).
Works in both bull and bear markets by following weekly trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6678_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for EMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    high_roll = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    low_roll = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD)
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
                
        # Determine trend from weekly EMA
        weekly_bullish = close[i] > ema_1w_aligned[i]
        weekly_bearish = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Donchian breakout signals
        breakout_up = (i > 0 and close[i] > high_roll[i-1])  # Current close above previous period's high
        breakout_down = (i > 0 and close[i] < low_roll[i-1])  # Current close below previous period's low
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_up and weekly_bullish and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif breakout_down and weekly_bearish and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals