#!/usr/bin/env python3
"""
exp_6958_1d_donchian20_1w_ema_vol_v2
Hypothesis: 1d Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation. 
Improved from failed exp_6950 by reducing signal size to 0.20, tightening volume threshold to 1.5x, 
adding ATR(14) stoploss at 2.0x, and ensuring proper warmup. Target: 50-150 total trades over 4 years 
(12-37/year) to avoid fee drag. Works in bull/bear by only taking breakouts aligned with weekly trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6958_1d_donchian20_1w_ema_vol_v2"
timeframe = "1d"
leverage = 1.0

# Parameters - tightened to reduce trade frequency
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5  # reduced from 2.0 to increase signal reliability
SIGNAL_SIZE = 0.20        # reduced from 0.25 to lower risk per trade
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0 # reduced from 2.5 for tighter risk control
MAX_HOLD_BARS = 40        # increased from 30 to allow trends to develop
EMA_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly EMA
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 with proper min_periods
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (1d) - align_htf_to_ltf handles shift(1) for completed bars only
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels with min_periods
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation with min_periods
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss with proper calculation
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr1 = pd.Series(high_low)
    tr2 = pd.Series(high_close)
    tr3 = pd.Series(low_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period - ensure all indicators are valid
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available (shouldn't happen with align_htf_to_ltf)
        if np.isnan(ema_1w_aligned[i]):
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
            
        # Volume confirmation - avoid division by zero or NaN
        vol_confirmed = False
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
        
        # Determine trend direction from weekly EMA50
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals aligned with weekly trend
        long_breakout = weekly_uptrend and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = weekly_downtrend and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals
</file>