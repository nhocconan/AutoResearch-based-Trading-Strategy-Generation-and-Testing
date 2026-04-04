#!/usr/bin/env python3
"""
exp_6531_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels as directional filter and volume confirmation.
Long when price > 1d EMA200 (bullish bias) and breaks above Donchian high with volume > 2.0x MA.
Short when price < 1d EMA200 (bearish bias) and breaks below Donchian low with volume > 2.0x MA.
Uses 1d EMA200 for trend bias and Camarilla levels only for exit logic (fade at R3/S3, breakout at R4/S4).
Designed for low-frequency, high-conviction trades targeting 75-200 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6531_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 200
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # volume must be 2.0x its 20-period MA for confirmation
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA200 and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to LTF (6h) with shift(1) for completed bars only
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
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
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
            continue
            
        # Long conditions: price > 1d EMA200 (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > ema_1d_aligned[i]  # price above 1d EMA200 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 1d EMA200 (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < ema_1d_aligned[i]  # price below 1d EMA200 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions:
        # 1. EMA reversal (trend change)
        # 2. Camarilla fade: long exits at R3, short exits at S3
        # 3. Camarilla breakout: reverse position if breaks R4/S4 with volume
        if position == 1:  # long position
            # Exit if price drops back below EMA200 (trend change)
            exit_long = close[i] < ema_1d_aligned[i]
            # Or if price drops to Camarilla R3 (fade level)
            exit_long = exit_long or close[i] <= camarilla_r3_aligned[i]
            # Reverse to short if breaks Camarilla R4 with volume (breakout failure)
            reverse_short = (close[i] >= camarilla_r4_aligned[i] and 
                           volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False)
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
            elif reverse_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                continue
        elif position == -1:  # short position
            # Exit if price rises back above EMA200 (trend change)
            exit_short = close[i] > ema_1d_aligned[i]
            # Or if price rises to Camarilla S3 (fade level)
            exit_short = exit_short or close[i] >= camarilla_s3_aligned[i]
            # Reverse to long if breaks Camarilla S4 with volume (breakout failure)
            reverse_long = (close[i] <= camarilla_s4_aligned[i] and 
                          volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False)
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
            elif reverse_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
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