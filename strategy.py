#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (13/8/5 SMAs) + Elder Ray (13-period bull/bear power) + volume confirmation
# Uses Williams Alligator for trend direction (jaw-teeth-lips alignment) and Elder Ray for momentum strength
# Volume spike (>2.0x 20-bar average) confirms participation
# ATR-based trailing stop via signal=0 when price retraces 30% of ATR from extreme
# Discrete sizing 0.25 to balance profit potential and fee drag; target 60-120 total trades over 4 years (15-30/year)
# Works in both bull/bear: Alligator catches trends, Elder Ray filters weak moves, volume ensures participation

name = "4h_WilliamsAlligator_ElderRay_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (13/8/5 SMAs) on 1d
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # 13-period SMA (jaw)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # 8-period SMA (teeth)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # 5-period SMA (lips)
    
    # Calculate Elder Ray (13-period bull/bear power) on 1d
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13  # Bull power = High - EMA13
    bear_power = low_1d - ema13   # Bear power = Low - EMA13
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long: Alligator aligned (jaws < teeth < lips) AND bull power > 0 AND volume spike
            if jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and bull_power_aligned[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short: Alligator reversed (jaws > teeth > lips) AND bear power < 0 AND volume spike
            elif jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and bear_power_aligned[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 30% of ATR from extreme
            if close[i] <= long_extreme - 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 30% of ATR from extreme
            if close[i] >= short_extreme + 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals