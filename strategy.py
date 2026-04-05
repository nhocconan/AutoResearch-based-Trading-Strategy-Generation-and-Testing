#!/usr/bin/env python3
"""
exp_7290_1d_donchian20_1w_ema_vol_v1
Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation.
In trending markets (price > 1w EMA): continuation breakouts in breakout direction.
In ranging markets (price near 1w EMA): mean reversion at Donchian extremes with volume confirmation.
Uses 1w EMA for trend regime and 1d volume for confirmation.
Designed for 1d timeframe to capture swings with ~7-25 trades/year (30-100 total over 4 years).
Works in both bull and bear markets by adapting to 1w EMA-defined trend regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7290_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 10  # 10 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for EMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
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
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on 1w EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        near_ema = np.abs(close[i] - ema_1w_aligned[i]) < (0.5 * atr[i])  # Within 0.5 ATR of EMA
        
        # Fade at extremes in ranging market (near EMA)
        fade_long = near_ema and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = near_ema and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market
        continuation_long = above_ema and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = below_ema and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
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