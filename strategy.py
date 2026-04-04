#!/usr/bin/env python3
"""
exp_6595_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with weekly Camarilla pivot direction and volume confirmation.
Uses 6h primary timeframe (target: 50-150 total trades over 4 years). Weekly Camarilla pivot levels
(S3/R3 for mean reversion, S4/R4 for breakout) provide institutional reference points that work in
both bull and bear markets. Volume confirmation ensures breakouts have conviction. Discrete sizing
(0.25) minimizes fee churn. Includes ATR-based stoploss.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6595_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Volume threshold for confirmation
SIGNAL_SIZE = 0.25      # 25% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5  # Stoploss at 2.5 * ATR
PIVOT_LOOKBACK = 5      # Bars to confirm pivot level respect

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3 = pivot + (range_1w * 1.1 / 4)
    s3 = pivot - (range_1w * 1.1 / 4)
    r4 = pivot + (range_1w * 1.1 / 2)
    s4 = pivot - (range_1w * 1.1 / 2)
    
    # Align to LTF (6h) with shift(1) for completed weeks only
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
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
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
                
        # Determine pivot-based bias
        # Price above R3: bullish bias (favor longs/breakouts)
        # Price below S3: bearish bias (favor shorts/breakdowns)
        # Between S3 and R3: neutral (wait for breakout)
        bullish_bias = close[i] > r3_aligned[i]
        bearish_bias = close[i] < s3_aligned[i]
        
        # Breakout conditions (continuation)
        # Long: break above R4 with volume
        # Short: break below S4 with volume
        long_breakout = close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1]
        short_breakout = close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1]
        
        # Volume confirmation
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Mean reversion conditions (fade at extremes)
        # Long: price touches S3 and starts reversing up
        # Short: price touches R3 and starts reversing down
        long_reversion = (close[i] <= s3_aligned[i] * 1.001 and  # touched S3
                          close[i] > close[i-1] and              # reversing up
                          close[i-1] <= s3_aligned[i-1])         # was at/below S3
        short_reversion = (close[i] >= r3_aligned[i] * 0.999 and # touched R3
                           close[i] < close[i-1] and             # reversing down
                           close[i-1] >= r3_aligned[i-1])        # was at/above R3
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif long_reversion:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_reversion:
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