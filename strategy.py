#!/usr/bin/env python3
"""
exp_6787_6h_donchian20_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels and volume confirmation.
Go long when price breaks above Donchian high + R3 pivot level with volume confirmation.
Go short when price breaks below Donchian low + S3 pivot level with volume confirmation.
Use 1d HTF for structural context - only take long trades when price > 1d VWAP (bull regime),
only short trades when price < 1d VWAP (bear regime). This avoids counter-trend whipsaws.
Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by aligning with 1d VWAP regime filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6787_6h_donchian20_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~5 days (6h bars)
PIVOT_LOOKBACK = 1  # use previous day's pivot levels

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivot and VWAP
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP for regime filter
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_array = vwap_1d.values
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous completed day's levels (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # first value has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    rng = prev_high - prev_low
    r3 = prev_close + (rng * 1.1 / 4)
    s3 = prev_close - (rng * 1.1 / 4)
    r4 = prev_close + (rng * 1.1 / 2)
    s4 = prev_close - (rng * 1.1 / 2)
    
    # Align HTF data to LTF (6h)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Regime filter: use 1d VWAP to determine bull/bear regime
        bull_regime = close[i] > vwap_1d_aligned[i]
        bear_regime = close[i] < vwap_1d_aligned[i]
        
        # Breakout signals with Camarilla confirmation
        # Long: price breaks above Donchian high AND above R3 level in bull regime
        long_breakout = bull_regime and (close[i] > highest_high[i]) and (close[i] > r3_aligned[i]) and vol_confirmed
        # Short: price breaks below Donchian low AND below S3 level in bear regime
        short_breakout = bear_regime and (close[i] < lowest_low[i]) and (close[i] < s3_aligned[i]) and vol_confirmed
        
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