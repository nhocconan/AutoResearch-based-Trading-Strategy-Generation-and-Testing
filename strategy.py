#!/usr/bin/env python3
"""
exp_6507_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1-week Camarilla pivot direction filter and volume confirmation.
Uses 1-week Camarilla pivot levels: long only when price > weekly R4 (bullish bias), short only when price < weekly S4 (bearish bias).
Donchian(20) breakout provides entry timing, volume confirmation filters weak breakouts.
Designed to work in both bull and bear markets by using weekly pivot extremes as regime filter and Donchian breakouts for momentum.
Target: 75-200 trades over 4 years (19-50/year). Uses 6h primary timeframe per experiment instructions.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6507_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
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
    
    # Load HTF data ONCE before loop - using 1w for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Camarilla pivot levels (R3, R4, S3, S4)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = H - L
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3_1w = pp_1w + range_1w * 1.1 / 2  # R3 = PP + (H-L)*1.1/2
    r4_1w = pp_1w + range_1w * 1.1      # R4 = PP + (H-L)*1.1
    s3_1w = pp_1w - range_1w * 1.1 / 2  # S3 = PP - (H-L)*1.1/2
    s4_1w = pp_1w - range_1w * 1.1      # S4 = PP - (H-L)*1.1
    
    # Align to LTF (6h) with shift(1) for completed bars only
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
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
        # Skip if weekly pivot data not available
        if np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]):
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above weekly R4 + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_regime = close[i] > r4_1w_aligned[i]      # price above weekly R4 (bullish bias)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below weekly S4 + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_regime = close[i] < s4_1w_aligned[i]     # price below weekly S4 (bearish bias)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: simple midpoint reversal
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
            if long_breakout and long_regime and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and short_regime and short_volume:
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