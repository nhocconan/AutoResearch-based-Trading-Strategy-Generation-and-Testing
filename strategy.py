#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for RSI and MA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly RSI (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Weekly 50-period SMA for trend
    sma50_1w = np.convolve(close_1w, np.ones(50)/50, mode='same')
    
    # Align weekly indicators to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(sma50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 50 and price above weekly SMA50 (bullish bias)
            long_cond = (rsi_1w_aligned[i] > 50) and (close[i] > sma50_1w_aligned[i])
            
            # Short: RSI < 50 and price below weekly SMA50 (bearish bias)
            short_cond = (rsi_1w_aligned[i] < 50) and (close[i] < sma50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI < 40 (momentum loss) or price below SMA50
            if (rsi_1w_aligned[i] < 40) or (close[i] < sma50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI > 60 (momentum loss) or price above SMA50
            if (rsi_1w_aligned[i] > 60) or (close[i] > sma50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly RSI > 50 indicates bullish momentum bias, < 50 bearish.
# Price relative to weekly SMA50 confirms trend alignment.
# Long when RSI > 50 and price above weekly SMA50.
# Short when RSI < 50 and price below weekly SMA50.
# Exits when RSI shows momentum exhaustion (RSI < 40 for longs, > 60 for shorts)
# or price crosses weekly SMA50 in opposite direction.
# Weekly timeframe filters noise, daily execution provides timely entries.
# Works in bull markets (follow RSI > 50) and bear markets (follow RSI < 50).
# Conservative position sizing (0.25) limits drawdown in volatile markets.
# Target: 20-60 trades over 4 years = 5-15/year to minimize fee decay.