#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(2) mean reversion with 1-week trend filter and volume confirmation.
# Uses contrarian RSI(2) for short-term reversals while filtering by weekly trend.
# In weekly uptrend (price > 200-week MA): long when RSI(2) < 10, exit when RSI(2) > 50.
# In weekly downtrend (price < 200-week MA): short when RSI(2) > 90, exit when RSI(2) < 50.
# Volume confirmation: current volume > 1.5x 20-period average to avoid low-liquidity signals.
# Target: 15-30 trades/year per side to stay within frequency limits and reduce fee drag.
name = "4h_RSI2_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 200-week simple moving average for trend filter
    def sma(arr, window):
        result = np.full_like(arr, np.nan)
        if len(arr) < window:
            return result
        for i in range(window-1, len(arr)):
            result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    sma_200w = sma(close_1w, 200)
    sma_200w_aligned = align_htf_to_ltf(prices, df_1w, sma_200w)
    
    # Calculate RSI(2) on 4h close
    def rsi(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period + 1:
            return result
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        result = 100 - (100 / (1 + rs))
        return result
    
    rsi_2 = rsi(close, 2)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(202, 20)  # Ensure 200-week MA and RSI(2) are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_200w_aligned[i]) or np.isnan(rsi_2[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma_200w_val = sma_200w_aligned[i]
        rsi_val = rsi_2[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend determination
        is_uptrend = price > sma_200w_val
        is_downtrend = price < sma_200w_val
        
        if position == 0:
            # Look for mean reversion entries in direction of weekly trend
            if is_uptrend and volume_confirmed:
                # In uptrend: long on extreme RSI(2) oversold
                if rsi_val < 10:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and volume_confirmed:
                # In downtrend: short on extreme RSI(2) overbought
                if rsi_val > 90:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI(2) returns to neutral (50) or trend changes
            if rsi_val > 50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) returns to neutral (50) or trend changes
            if rsi_val < 50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals