#!/usr/bin/env python3
"""
4h RSI Divergence + Volume Spike + 12h EMA Trend Filter
Uses RSI(14) divergence with price action, confirmed by volume spikes and 12h EMA trend.
Designed for low trade frequency with high win rate in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily timeframe
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # RSI divergence detection (lookback 3 periods)
    rsi_rising = rsi_14_aligned > np.roll(rsi_14_aligned, 1)
    price_rising = close > np.roll(close, 1)
    rsi_falling = rsi_14_aligned < np.roll(rsi_14_aligned, 1)
    price_falling = close < np.roll(close, 1)
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    bull_div = (price_falling & (close < np.roll(close, 2))) & \
               (rsi_rising & (rsi_14_aligned > np.roll(rsi_14_aligned, 2)))
    # Bearish divergence: price makes higher high, RSI makes lower high
    bear_div = (price_rising & (close > np.roll(close, 2))) & \
               (rsi_falling & (rsi_14_aligned < np.roll(rsi_14_aligned, 2)))
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_14_aligned[i]
        ema_trend = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: bullish divergence with volume spike and above 12h EMA
            if (bull_div[i] and 
                volume_spike[i] and 
                price > ema_trend and
                rsi_val < 40):  # additional filter for oversold condition
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: bearish divergence with volume spike and below 12h EMA
            elif (bear_div[i] and 
                  volume_spike[i] and 
                  price < ema_trend and
                  rsi_val > 60):  # additional filter for overbought condition
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions: bearish divergence or RSI overbought
            if bear_div[i] or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions: bullish divergence or RSI oversold
            if bull_div[i] or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_RSI_Divergence_Volume_Spike_12hEMA34"
timeframe = "4h"
leverage = 1.0