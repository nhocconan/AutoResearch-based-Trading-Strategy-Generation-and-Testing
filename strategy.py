#!/usr/bin/env python3
name = "1d_Wick_Reversal_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 21  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate daily body and wicks
        body = np.abs(close[i] - open_[i])
        lower_wick = np.minimum(open_[i], close[i]) - low[i]
        upper_wick = high[i] - np.maximum(open_[i], close[i])
        
        # Volume spike: current volume > 1.5 * 20-day average volume
        vol_ma20 = np.mean(np.maximum(1, volume[max(0, i-19):i+1])) if i >= 19 else volume[i]
        volume_spike = volume[i] > 1.5 * vol_ma20
        
        if position == 0:
            # Long reversal: long lower wick, small body, volume spike, above weekly EMA
            if (lower_wick > 2 * body and  # Long lower wick at least 2x body
                body < (high[i] - low[i]) * 0.3 and  # Small body (<30% of range)
                volume_spike and
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short reversal: long upper wick, small body, volume spike, below weekly EMA
            elif (upper_wick > 2 * body and  # Long upper wick at least 2x body
                  body < (high[i] - low[i]) * 0.3 and  # Small body (<30% of range)
                  volume_spike and
                  close[i] < ema21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly EMA or opposite signal
            if close[i] < ema21_1w_aligned[i] or (upper_wick > 2 * body and volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly EMA or opposite signal
            if close[i] > ema21_1w_aligned[i] or (lower_wick > 2 * body and volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals