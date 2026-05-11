#!/usr/bin/env python3
"""
1d_Pullback_Reversal_v1
Hypothesis: Daily pullback reversals after sharp moves. After a 5% daily drop, wait for RSI oversold (<30) and reversal candle (bullish engulfing) to go long. After 5% daily rise, wait for RSI overbought (>70) and bearish engulfing to short. Uses 1w EMA50 trend filter to align with weekly trend. Designed for low frequency: only trades on extreme daily moves with confirmation. Works in bull markets by buying dips in uptrend, and in bear markets by selling rallies in downtrend.
"""

name = "1d_Pullback_Reversal_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily data for calculations
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily returns for extreme move detection
    daily_ret = (close / np.roll(close, 1)) - 1
    daily_ret[0] = 0  # first bar
    
    # RSI(14) on daily close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bullish/bearish engulfing patterns
    bullish_engulf = (close > open_) & (open_ > np.roll(close, 1)) & (close > np.roll(open_, 1))
    bearish_engulf = (close < open_) & (open_ < np.roll(close, 1)) & (close < np.roll(open_, 1))
    
    # Weekly EMA50 trend filter
    weekly_close = df_1w['close'].values
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for RSI
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(bullish_engulf[i]) or np.isnan(bearish_engulf[i]) or
            np.isnan(daily_ret[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: daily drop >5%, RSI oversold, bullish engulfing, above weekly EMA
            if (daily_ret[i] < -0.05 and 
                rsi[i] < 30 and 
                bullish_engulf[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily rise >5%, RSI overbought, bearish engulfing, below weekly EMA
            elif (daily_ret[i] > 0.05 and 
                  rsi[i] > 70 and 
                  bearish_engulf[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite extreme move or loss of trend alignment
            if position == 1:
                # Exit long: daily rise >5% or price below weekly EMA
                if (daily_ret[i] > 0.05 or close[i] < ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: daily drop >5% or price above weekly EMA
                if (daily_ret[i] < -0.05 or close[i] > ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals