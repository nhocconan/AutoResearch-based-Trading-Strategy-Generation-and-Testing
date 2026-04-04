#!/usr/bin/env python3
"""
exp_6700_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1-day EMA trend filter and volume confirmation.
In trending markets (price > 1d EMA50), buy breakouts above 20-period high; in ranging markets (price near 1d EMA50), fade reversals from 20-period lows.
Volume confirmation filters false breakouts. ATR-based stoploss manages risk.
Designed for 4h timeframe to capture medium-term swings while minimizing fee drag (~20-50 trades/year expected).
Works in both bull (trend continuation) and bear (mean reversion near EMA) markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6700_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 6  # ~1 day (4h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    high_ma = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    low_ma = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
        
        # Skip if indicators not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine market regime based on 1d EMA50
        # Strong uptrend: price well above EMA50
        # Strong downtrend: price well below EMA50
        # Ranging/transition: price near EMA50
        ema_diff_pct = (close[i] - ema_1d_aligned[i]) / ema_1d_aligned[i] * 100
        strong_uptrend = ema_diff_pct > 2.0   # price > 2% above EMA50
        strong_downtrend = ema_diff_pct < -2.0  # price > 2% below EMA50
        ranging_market = abs(ema_diff_pct) <= 2.0  # price within 2% of EMA50
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Breakout signals (in trending markets)
        long_breakout = strong_uptrend and (close[i] > high_ma[i]) and vol_confirmed
        short_breakout = strong_downtrend and (close[i] < low_ma[i]) and vol_confirmed
        
        # Mean reversion signals (in ranging markets)
        long_mean_revert = ranging_market and (close[i] < low_ma[i]) and vol_confirmed
        short_mean_revert = ranging_market and (close[i] > high_ma[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout or long_mean_revert:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout or short_mean_revert:
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