#!/usr/bin/env python3
"""
exp_6499_6h_donchian20_12h_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 12h Camarilla pivot direction filter and volume confirmation.
Uses 12h Camarilla pivot levels (R3/S3, R4/S4) to determine institutional bias: 
- Long when price > R3 and breaks above Donchian high with volume
- Short when price < S3 and breaks below Donchian low with volume
Camarilla pivots from higher timeframe (12h) provide structure that works in both bull and bear markets by identifying key reversal/continuation levels.
Volume confirmation filters weak breakouts. Target: 75-150 trades over 4 years (19-37/year).
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6499_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # Higher threshold for stricter volume confirmation
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla pivot calculation: based on previous period's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    rng = high_12h - low_12h
    r3 = close_12h + 1.0 * rng
    s3 = close_12h - 1.0 * rng
    r4 = close_12h + 1.5 * rng
    s4 = close_12h - 1.5 * rng
    
    # Align to LTF (6h) with shift(1) for completed bars only
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            continue
            
        # Long conditions: price > R3 (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > r3_aligned[i]  # price above R3 indicates bullish institutional bias
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < S3 (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < s3_aligned[i]  # price below S3 indicates bearish institutional bias
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: pivot-based exits
        if position == 1:  # long position
            # Exit if price drops below S3 (bearish bias) or breaks below Donchian low
            exit_long = close[i] < s3_aligned[i] or close[i] < donchian_low[i-1]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above R3 (bullish bias) or breaks above Donchian high
            exit_short = close[i] > r3_aligned[i] or close[i] > donchian_high[i-1]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>