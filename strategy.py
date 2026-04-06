#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Exponential Moving Average crossover with 1-week RSI momentum filter and volume confirmation.
# EMA crossover provides trend-following signals with reduced whipsaw.
# RSI filter avoids counter-trend entries in overbought/oversold conditions.
# Volume confirmation ensures institutional participation.
# Designed for 1d timeframe to target 30-100 trades over 4 years with low frequency.

name = "1d_ema20_50_rsi1w_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day EMA(20) and EMA(50) for trend signals
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    
    # 1-week RSI(14) for momentum filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # RSI calculation
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    
    for i in range(14, len(close_1w)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 14, 4)  # EMA50 needs 50, RSI needs 14, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: EMA crossover down or stoploss
            if (ema20[i] < ema50[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: EMA crossover up or stoploss
            if (ema20[i] > ema50[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with EMA crossover and RSI filter
            if volume_filter:
                # Long: EMA20 crosses above EMA50 and RSI not overbought
                if (ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1] and 
                    rsi_aligned[i] < 70):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: EMA20 crosses below EMA50 and RSI not oversold
                elif (ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1] and 
                      rsi_aligned[i] > 30):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals