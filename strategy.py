#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band width contraction + daily RSI extremes on 12h timeframe.
# Uses weekly BB width to identify low volatility (squeeze) regime and daily RSI for mean reversion entries.
# Enters when price touches Bollinger Bands during squeeze with RSI extreme reversal.
# Designed for low trade frequency to avoid fee drag, works in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20_1w = np.full(len(close_1w), np.nan)
    for i in range(bb_period, len(close_1w)):
        sma_20_1w[i] = np.mean(close_1w[i-bb_period:i])
    
    std_20_1w = np.full(len(close_1w), np.nan)
    for i in range(bb_period, len(close_1w)):
        std_20_1w[i] = np.std(close_1w[i-bb_period:i])
    
    bb_upper_1w = sma_20_1w + bb_std * std_20_1w
    bb_lower_1w = sma_20_1w - bb_std * std_20_1w
    bb_width_1w = (bb_upper_1w - bb_lower_1w) / sma_20_1w  # Normalized width
    
    # Align weekly BB data to 12h timeframe
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    bb_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_upper_1w)
    bb_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_lower_1w)
    bb_width_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_width_1w)
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    # Initial average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 12h ATR for stop loss
    tr_12h_1 = high - low
    tr_12h_2 = np.abs(high - np.roll(close, 1))
    tr_12h_3 = np.abs(low - np.roll(close, 1))
    tr_12h_1[0] = high[0] - low[0]
    tr_12h_2[0] = np.abs(high[0] - close[0])
    tr_12h_3[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr_12h_1, np.maximum(tr_12h_2, tr_12h_3))
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, rsi_period, 20)  # need weekly BB, daily RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(sma_20_1w_aligned[i]) or np.isnan(bb_upper_1w_aligned[i]) or 
            np.isnan(bb_lower_1w_aligned[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: weekly BB width below 20th percentile (low volatility)
        # Calculate percentile using historical values up to current point
        if i >= 50:  # need sufficient history for percentile
            width_history = bb_width_1w_aligned[:i+1]
            width_history_valid = width_history[~np.isnan(width_history)]
            if len(width_history_valid) >= 20:
                width_percentile = np.percentile(width_history_valid, 20)
                squeeze = bb_width_1w_aligned[i] <= width_percentile
            else:
                squeeze = False
        else:
            squeeze = False
        
        # RSI extreme conditions
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Price touching Bollinger Bands
        touch_upper = close[i] >= bb_upper_1w_aligned[i]
        touch_lower = close[i] <= bb_lower_1w_aligned[i]
        
        if position == 0:
            # Long entry: squeeze + RSI oversold + touch lower band
            if squeeze and rsi_oversold and touch_lower:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze + RSI overbought + touch upper band
            elif squeeze and rsi_overbought and touch_upper:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses above middle band or RSI exits oversold
            if close[i] >= sma_20_1w_aligned[i] or rsi_1d_aligned[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below middle band or RSI exits overbought
            if close[i] <= sma_20_1w_aligned[i] or rsi_1d_aligned[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BBW_Squeeze_RSI_Extreme"
timeframe = "12h"
leverage = 1.0