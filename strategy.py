# 1) Hypothesis
# The strategy uses 1-day Exponential Moving Average (EMA) crossovers as the primary trend filter
# combined with 1-week Relative Strength Index (RSI) extremes to capture momentum in both bull and bear markets.
# Long entries occur when EMA(9) crosses above EMA(21) and weekly RSI is below 70 (avoiding overbought).
# Short entries occur when EMA(9) crosses below EMA(21) and weekly RSI is above 30 (avoiding oversold).
# Exits are triggered by the opposite EMA crossover.
# Position sizing is fixed at 0.25 to manage risk and reduce fee churn.
# This approach aims to capture trends while avoiding extreme readings, suitable for both bullish and bearish regimes.

# 2) Implementation
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get daily data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(9) and EMA(21)
    ema_9_1d = pd.Series(close_1d).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMAs to lower timeframe (1d candles)
    ema_9_aligned = align_htf_to_ltf(prices, df_1d, ema_9_1d)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    
    # Align RSI to lower timeframe (1d candles)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_9_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or
            np.isnan(rsi_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # EMA crossover signals
        ema_cross_up = ema_9_aligned[i] > ema_21_aligned[i]
        ema_cross_down = ema_9_aligned[i] < ema_21_aligned[i]
        
        # RSI conditions: avoid extremes
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Long conditions: bullish crossover + not overbought
        long_condition = ema_cross_up and rsi_not_overbought
        
        # Short conditions: bearish crossover + not oversold
        short_condition = ema_cross_down and rsi_not_oversold
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite EMA crossover
        elif position == 1 and ema_cross_down:
            signals[i] = 0.0
            position = 0
        elif position == -1 and ema_cross_up:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA9_21_1wRSI_Crossover"
timeframe = "1d"
leverage = 1.0