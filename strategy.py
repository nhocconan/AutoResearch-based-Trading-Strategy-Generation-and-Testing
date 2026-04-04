#!/usr/bin/env python3
"""
Experiment #2691: 6h Camarilla pivot from 1d + volume confirmation + ATR stop
HYPOTHESIS: 6h mean reversion at Camarilla R3/S3 levels with 1d trend filter and volume confirmation
captures institutional fade of overextended moves. Works in bull/bear via 1d EMA50 trend alignment.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2691_6h_camarilla_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate previous day's Camarilla levels (using prior 1d bar)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We shift by 1 to use only completed 1d bars (no look-ahead)
    shift_idx = np.arange(len(close_1d)) - 1
    valid = shift_idx >= 0
    ph = np.full_like(high_1d, np.nan)
    pl = np.full_like(low_1d, np.nan)
    pc = np.full_like(close_1d, np.nan)
    ph[valid] = high_1d[shift_idx[valid]]
    pl[valid] = low_1d[shift_idx[valid]]
    pc[valid] = close_1d[shift_idx[valid]]
    
    # Camarilla multipliers
    camarilla_mult = 1.1
    r3 = pc + ((ph - pl) * camarilla_mult / 4)
    s3 = pc - ((ph - pl) * camarilla_mult / 4)
    r4 = pc + ((ph - pl) * camarilla_mult / 2)
    s4 = pc - ((ph - pl) * camarilla_mult / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Volume MA(20), ATR(14) for stoploss ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # True Range and ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Stoploss: 2*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit at R4 (if long from S3) or mean reversion
                elif price >= r4_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit at S4 (if short from R3) or mean reversion
                elif price <= s4_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            # Long entry: price at S3 support with uptrend on 1d (fade extreme)
            if trend_bias > 0 and price <= s3_6h[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price at R3 resistance with downtrend on 1d (fade extreme)
            elif trend_bias < 0 and price >= r3_6h[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals