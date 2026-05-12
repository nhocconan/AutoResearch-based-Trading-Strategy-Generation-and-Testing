#!/usr/bin/env python3
name = "12h_EngulfingReversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bullish and bearish engulfing patterns
    bullish_engulf = (close > open_) & (close > np.roll(open_, 1)) & (open_ < np.roll(close, 1))
    bearish_engulf = (close < open_) & (close < np.roll(open_, 1)) & (open_ > np.roll(close, 1))
    
    # Volume confirmation: volume > 1.5 * average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish engulfing + volume confirmation + 1d uptrend
            if bullish_engulf[i] and vol_confirm[i] and (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing + volume confirmation + 1d downtrend
            elif bearish_engulf[i] and vol_confirm[i] and (close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish engulfing or trend reversal
            if bearish_engulf[i] or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish engulfing or trend reversal
            if bullish_engulf[i] or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals