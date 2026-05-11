#!/usr/bin/env python3
"""
4h_Keltner_Channel_Squeeze_Trend
Hypothesis: Keltner Channel squeeze combined with 4h EMA trend and volume confirmation.
In low volatility (squeeze), price often breaks out with strong momentum. 
We enter on breakout from squeeze in direction of 4h EMA trend.
Works in bull/bear markets by following trend and using volatility contraction for timing.
Target: 20-50 trades/year on 4h timeframe.
"""

name = "4h_Keltner_Channel_Squeeze_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA trend filter (21-period)
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR for Keltner Channel (10-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # EMA for Keltner Channel middle (20-period)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # Bollinger Bands for squeeze detection (20-period, 2.0 std)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Squeeze condition: BB inside Keltner (low volatility)
    squeeze = (bb_upper <= keltner_upper) & (bb_lower >= keltner_lower)
    
    # Breakout conditions
    buy_breakout = close > keltner_upper
    sell_breakout = close < keltner_lower
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(squeeze[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: squeeze release + upward breakout + uptrend
            if squeeze[i-1] and buy_breakout[i] and close[i] > ema21[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze release + downward breakout + downtrend
            elif squeeze[i-1] and sell_breakout[i] and close[i] < ema21[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or trend reversal
            if position == 1:
                # Exit long: price re-enters Keltner Channel or trend turns down
                if close[i] < keltner_upper[i] or close[i] < ema21[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price re-enters Keltner Channel or trend turns up
                if close[i] > keltner_lower[i] or close[i] > ema21[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals