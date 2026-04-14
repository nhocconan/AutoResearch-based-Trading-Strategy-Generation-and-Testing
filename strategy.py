#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy with 1w RSI trend filter and 1d Williams %R mean reversion
# 1w RSI > 50 indicates bullish long-term trend (favor longs), < 50 indicates bearish (favor shorts)
# 1d Williams %R below -80 indicates oversold, above -20 indicates overbought
# Combines trend filter with mean reversion to avoid counter-trend trades in strong moves
# Uses weekly trend context to improve daily mean reversion edges in both bull and bear markets
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE for RSI
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w RSI (14 periods)
    rsi_len = 14
    close_1w = df_1w['close'].values
    
    # Price changes
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Smoothed averages
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    
    # Relative Strength
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 0, rs)
    
    # RSI
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align RSI to 1d timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 1d Williams %R (14 periods)
    willr_len = 14
    highest_high = pd.Series(high).rolling(window=willr_len, min_periods=willr_len).max().values
    lowest_low = pd.Series(low).rolling(window=willr_len, min_periods=willr_len).min().values
    
    # Williams %R
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, rsi_len, willr_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(willr[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w RSI > 50 bullish, < 50 bearish
        bullish_trend = rsi_1w_aligned[i] > 50
        bearish_trend = rsi_1w_aligned[i] < 50
        
        # Mean reversion signals from Williams %R
        oversold = willr[i] < -80
        overbought = willr[i] > -20
        
        if position == 0:
            # Enter long: bullish weekly trend + daily oversold
            if bullish_trend and oversold:
                position = 1
                signals[i] = position_size
            # Enter short: bearish weekly trend + daily overbought
            elif bearish_trend and overbought:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (mean reversion complete) 
            # or weekly trend turns bearish
            if willr[i] > -50 or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (mean reversion complete) 
            # or weekly trend turns bullish
            if willr[i] < -50 or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wRSI_1dWR_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0