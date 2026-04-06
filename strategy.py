#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour RSI(2) Extreme Reversal with Trend Filter.
# Uses daily 200-period EMA as trend filter (long only above EMA200, short only below).
# Enters on RSI(2) extreme readings (<10 for long, >90 for short) with volume confirmation.
# Exits on RSI(2) crossing above/below 50 or stoploss at 2x ATR.
# Works in both bull and bear markets by following trend direction.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_rsi2_extreme_trend_filt_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = np.full(len(close_1d), np.nan)
    
    # Calculate EMA200 on daily close
    ema200_1d[199] = np.mean(close_1d[:200])
    for i in range(200, len(close_1d)):
        ema200_1d[i] = (close_1d[i] * 2/201) + (ema200_1d[i-1] * (1 - 2/201))
    
    # Align daily EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(2) calculation
    rsi = np.full(n, np.nan)
    if n >= 3:
        change = np.diff(close, prepend=close[0])
        gain = np.where(change > 0, change, 0.0)
        loss = np.where(change < 0, -change, 0.0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[2] = np.mean(gain[1:3])
        avg_loss[2] = np.mean(loss[1:3])
        
        for i in range(3, n):
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if EMA200 or volume data not available
        if np.isnan(ema200_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (rsi[i] >= 50 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (rsi[i] <= 50 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: RSI(2) < 10 and price above daily EMA200
                if (rsi[i] < 10 and close[i] > ema200_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: RSI(2) > 90 and price below daily EMA200
                elif (rsi[i] > 90 and close[i] < ema200_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals