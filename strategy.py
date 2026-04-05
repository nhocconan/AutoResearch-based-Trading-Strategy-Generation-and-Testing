#!/usr/bin/env python3
"""
exp_7227_6h_donchian20_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot continuation logic.
In trending markets, breakouts above R4 or below S4 continue.
In ranging markets, reversals at R3/S3 with volume confirmation.
Uses weekly trend filter (price > weekly EMA200) to avoid counter-trend trades.
Designed for 6h timeframe to achieve 50-150 trades over 4 years (12-37/year).
Works in bull/bear via weekly EMA200 trend filter and Camarilla structure.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7227_6h_donchian20_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
CAMARILLA_PERIOD = 1  # daily
EMA_TREND_PERIOD = 200  # weekly
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Typical price for pivot
    pp = (high_1d + low_1d + close_1d) / 3.0
    r = (high_1d - low_1d) * 1.1 / 2.0
    
    # Camarilla levels
    r3 = pp + r * 1.1
    s3 = pp - r * 1.1
    r4 = pp + r * 1.5
    s4 = pp - r * 1.5
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Align HTF data to LTF (6h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
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
    start = max(DONCHIAN_PERIOD, CAMARILLA_PERIOD, EMA_TREND_PERIOD//4, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(r3_aligned[i]):
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
        
        # Trend filter: only trade with weekly EMA200 trend
        above_weekly_ema = close[i] > ema_200_aligned[i]
        below_weekly_ema = close[i] < ema_200_aligned[i]
        
        # Camarilla-based signals
        # Breakout continuation: break R4/S4 with volume
        breakout_long = (close[i] > r4_aligned[i]) and vol_confirmed
        breakout_short = (close[i] < s4_aligned[i]) and vol_confirmed
        
        # Mean reversion fade: reverse at R3/S3 with volume
        fade_long = (close[i] < s3_aligned[i]) and vol_confirmed
        fade_short = (close[i] > r3_aligned[i]) and vol_confirmed
        
        # Entry logic: only if flat
        if position == 0:
            # In uptrend: take breakout longs and fade shorts at R3
            if above_weekly_ema:
                if breakout_long:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif fade_short:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            # In downtrend: take breakout shorts and fade longs at S3
            elif below_weekly_ema:
                if breakout_short:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif fade_long:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            # In ranging (near weekly EMA): fade both extremes
            else:
                if fade_long:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif fade_short:
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

</think>