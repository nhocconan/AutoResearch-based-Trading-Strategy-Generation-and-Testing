#!/usr/bin/env python3
"""
exp_6507_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot continuation and volume confirmation.
In bear markets (2025+), price often respects Camarilla levels from higher timeframe.
Breakouts above R4 or below S4 with volume continuation signal strong momentum in trend direction.
Breakouts at R3/S3 with volume often fade (mean reversion). Uses 1d pivot for structure, 6h for timing.
Uses volume spike to confirm institutional participation. Works in both bull/bear by following momentum.
Target: 75-200 trades over 4 years (19-50/year). Uses 6h primary timeframe per experiment instructions.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6507_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # use previous day's pivot
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # volume must be 2x its 20-period MA for confirmation
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r3 = pivot + (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    # Support levels
    s3 = pivot - (range_1d * 1.1 / 2)
    s4 = pivot - (range_1d * 1.1)
    
    # Align to LTF (6h) with shift(1) for completed bars only
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if pivot data not available
        if np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
            continue
            
        # Long breakout conditions
        long_breakout_r4 = close[i] > donchian_high[i-1] and close[i] > r4_aligned[i]  # break above Donchian AND R4
        long_breakout_r3 = close[i] > donchian_high[i-1] and close[i] > r3_aligned[i] and close[i] <= r4_aligned[i]  # break above Donchian but at R3
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short breakout conditions
        short_breakout_s4 = close[i] < donchian_low[i-1] and close[i] < s4_aligned[i]  # break below Donchian AND S4
        short_breakout_s3 = close[i] < donchian_low[i-1] and close[i] < s3_aligned[i] and close[i] >= s4_aligned[i]  # break below Donchian but at S3
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: midpoint reversal or opposite Donchian break
        if position == 1:  # long position
            # Exit if price drops below midpoint of channel
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks below Donchian low (strong reversal)
            exit_long = exit_long or close[i] < donchian_low[i-1]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above midpoint of channel
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks above Donchian high (strong reversal)
            exit_short = exit_short or close[i] > donchian_high[i-1]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            # Long: breakout at R4 (continuation) OR at R3 with volume (fade play - less reliable)
            if long_breakout_r4 and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            # Short: breakout at S4 (continuation) OR at S3 with volume (fade play - less reliable)
            elif short_breakout_s4 and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals