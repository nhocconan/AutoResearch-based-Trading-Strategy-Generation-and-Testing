#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Williams %R and 1w RSI divergence filter
# Williams %R identifies overbought/oversold conditions on daily chart
# RSI on weekly chart confirms momentum divergence (bullish/bearish)
# Strategy enters on mean reversion in ranging markets and trend continuation in trending markets
# Uses Williams %R for entry timing and weekly RSI for regime filter to avoid whipsaws
# Designed to work in both bull and bear markets by adapting to volatility regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14 periods)
    wr_length = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=wr_length, min_periods=wr_length).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=wr_length, min_periods=wr_length).min().values
    # Avoid division by zero
    denom = highest_high - lowest_low
    wr = np.where(denom != 0, -100 * (highest_high - df_1d['close'].values) / denom, -50)
    # Williams %R ranges from -100 to 0, where -20-0 is overbought, -100--80 is oversold
    
    # Align Williams %R to 6h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Load 1w data ONCE for RSI
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI (14 periods) on weekly
    rsi_length = 14
    delta = pd.Series(df_1w['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Smoothed averages
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, wr_length, rsi_length)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Williams %R signals
        wr_oversold = wr_aligned[i] <= -80
        wr_overbought = wr_aligned[i] >= -20
        
        # RSI regime filter: 
        # RSI < 40 = bearish bias (favor shorts on OBS)
        # RSI > 60 = bullish bias (favor longs on OS)
        # 40 <= RSI <= 60 = neutral (mean reversion both ways)
        rsi_bullish = rsi_aligned[i] > 60
        rsi_bearish = rsi_aligned[i] < 40
        rsi_neutral = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        if position == 0:
            # Enter long: Williams %R oversold AND (RSI bullish OR neutral)
            if wr_oversold and (rsi_bullish or rsi_neutral):
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought AND (RSI bearish OR neutral)
            elif wr_overbought and (rsi_bearish or rsi_neutral):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R overbought OR RSI turns bearish
            if wr_overbought or rsi_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R oversold OR RSI turns bullish
            if wr_oversold or rsi_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dWR_1wRSI_Divergence_v1"
timeframe = "6h"
leverage = 1.0