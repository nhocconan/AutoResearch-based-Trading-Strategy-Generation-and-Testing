#!/usr/bin/env python3
"""
exp_6512_12h_donchian20_1d_pivot_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1d Camarilla pivot direction filter and volume confirmation.
Uses 1d Camarilla pivot levels (R3, S3, R4, S4) to determine bias: long when price > R3, short when price < S3.
Donchian(20) breakout provides entry timing in the direction of 1d pivot bias, volume confirmation filters weak breakouts.
Designed to work in both bull and bear markets by using 1d Camarilla levels as structural support/resistance.
Target: 50-150 trades over 4 years (12-37/year). Uses 12h primary timeframe per experiment instructions.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6512_12h_donchian20_1d_pivot_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.8  # volume must be 1.8x its 20-period MA
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = close + (high - low) * 1.1/2, R3 = close + (high - low) * 1.1/4
    #                S3 = close - (high - low) * 1.1/4, S4 = close - (high - low) * 1.1/2
    camarilla_high = high_1d - low_1d
    r4 = close_1d + camarilla_high * 1.1 / 2
    r3 = close_1d + camarilla_high * 1.1 / 4
    s3 = close_1d - camarilla_high * 1.1 / 4
    s4 = close_1d - camarilla_high * 1.1 / 2
    
    # Align to LTF (12h) with shift(1) for completed bars only
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
            
        # Determine bias from 1d Camarilla levels
        long_bias = close[i] > r3_aligned[i]  # price above R3 = bullish bias
        short_bias = close[i] < s3_aligned[i]  # price below S3 = bearish bias
        
        # Long conditions: price breaks above Donchian HIGH + long bias + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + short bias + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: pivot level reversal
        if position == 1:  # long position
            # Exit if price drops below S3 (bearish bias)
            exit_long = close[i] < s3_aligned[i]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above R3 (bullish bias)
            exit_short = close[i] > r3_aligned[i]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_bias and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and short_bias and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals