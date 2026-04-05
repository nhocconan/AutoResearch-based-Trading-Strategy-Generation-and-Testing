#!/usr/bin/env python3
"""
exp_6938_1d_donchian20_1w_ema_vol_v2
Hypothesis: 1d Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation, 
adjusted for higher trade frequency by relaxing volume threshold slightly and adding 
choppiness regime filter to avoid whipsaws in ranging markets. 
Targets 50-150 total trades over 4 years (12-38/year) by allowing entries in both 
trending and ranging regimes with appropriate filters.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6938_1d_donchian20_1w_ema_vol_v2"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5  # Reduced from 2.0 to increase trade frequency
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 30  # ~1.5 months (1d bars)
EMA_PERIOD = 50
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # Above this = ranging market

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly EMA
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50
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
    
    # Choppiness Index for regime detection
    atr_rolling = tr.rolling(window=CHOPPINESS_PERIOD, min_periods=CHOPPINESS_PERIOD).mean().values
    highest_high_rolling = pd.Series(high).rolling(window=CHOPPINESS_PERIOD, min_periods=CHOPPINESS_PERIOD).max().values
    lowest_low_rolling = pd.Series(low).rolling(window=CHOPPINESS_PERIOD, min_periods=CHOPPINESS_PERIOD).min().values
    
    # Avoid division by zero
    range_max_min = highest_high_rolling - lowest_low_rolling
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    
    chop = 100 * np.log10(atr_rolling * CHOPPINESS_PERIOD / range_max_min) / np.log10(CHOPPINESS_PERIOD)
    chopping_market = chop > CHOPPINESS_THRESHOLD  # True when ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD, CHOPPINESS_PERIOD) + 1
    
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
        
        # Determine trend direction from weekly EMA50
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals with regime adaptation
        # In trending markets: follow weekly trend
        # In ranging markets: trade both directions but with stricter volume
        if chopping_market:
            # Ranging market: look for breakouts in either direction with volume
            long_breakout = (close[i] > highest_high[i]) and vol_confirmed
            short_breakout = (close[i] < lowest_low[i]) and vol_confirmed
        else:
            # Trending market: follow weekly trend direction
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