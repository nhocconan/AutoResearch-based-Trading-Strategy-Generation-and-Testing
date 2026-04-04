#!/usr/bin/env python3
"""
Experiment #2891: 6h Williams %R + 1d Trend Filter + Volume Confirmation
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h.
Only take trades in direction of 1d EMA50 trend: long when price > EMA50, short when price < EMA50.
Volume spike (>1.8x 20-period average) confirms momentum. This avoids counter-trend
whipsaws in ranging markets while capturing strong trending moves. 6h timeframe
provides sufficient signal quality with controlled trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2891_6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily closes
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full(n, np.nan)
    denominator = highest_high - lowest_low
    mask = denominator != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / denominator[mask]) * -100
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: Williams %R returns to neutral zone (between -20 and -80)
            if position_side > 0:  # Long
                if williams_r[i] > -20:  # Exited overbought
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if williams_r[i] < -80:  # Exited oversold
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Get 1d trend filter: price vs EMA50
            price_above_ema = price > ema50_1d_aligned[i]
            
            # Long entry: Williams %R oversold (< -80) AND bullish trend (price > EMA50)
            if williams_r[i] < -80 and price_above_ema:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Williams %R overbought (> -20) AND bearish trend (price < EMA50)
            elif williams_r[i] > -20 and not price_above_ema:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>