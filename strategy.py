#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Exponential Moving Average (EMA) crossover with weekly trend filter and volume confirmation
# Uses EMA(20) and EMA(50) crossovers on 1d for entry signals
# Weekly EMA(20) as trend filter (only long when weekly close > weekly EMA20, short when weekly close < weekly EMA20)
# Volume confirmation > 1.5x 20-period EMA on 1d to reduce false signals
# Designed for 15-30 trades/year with clear momentum logic
# Works in bull markets via bullish crossover + uptrend and in bear markets via bearish crossover + downtrend
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(20) and EMA(50) for 1d
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned weekly EMA20
        ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)[i]
        
        if np.isnan(ema20_1w_aligned) or np.isnan(vol_ma[i]) or np.isnan(ema20[i]) or np.isnan(ema50[i]):
            continue
        
        # Trend filter: only long in uptrend, only short in downtrend
        uptrend = close_1w[-1] > ema20_1w_aligned if len(close_1w) > 0 else False  # Use last known weekly close
        downtrend = close_1w[-1] < ema20_1w_aligned if len(close_1w) > 0 else False
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Golden Cross: EMA20 crosses above EMA50 + uptrend + volume confirmation
        if position == 0 and ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1] and uptrend and volume_confirm:
            position = 1
            signals[i] = position_size
        # Death Cross: EMA20 crosses below EMA50 + downtrend + volume confirmation
        elif position == 0 and ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1] and downtrend and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite crossover or loss of trend
        elif position != 0:
            if position == 1 and (ema20[i] < ema50[i] or not uptrend):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (ema20[i] > ema50[i] or not downtrend):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_EMA_Crossover_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0