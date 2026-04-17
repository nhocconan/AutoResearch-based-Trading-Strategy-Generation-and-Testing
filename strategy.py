#!/usr/bin/env python3
"""
1d_1w_RSI_Divergence_Momentum_V1
Hypothesis: On daily timeframe, buy when RSI(14) shows bullish divergence with price (higher low in RSI while price makes lower low) and weekly close > weekly open, sell when RSI shows bearish divergence (lower high in RSI while price makes higher high) and weekly close < weekly open. Uses volume confirmation (>1.3x average volume) to filter weak signals. Designed for low frequency (target 15-25 trades/year) to work in both bull and bear markets by capturing momentum reversals at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Data (HTF for trend and divergence) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly RSI (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        if len(prices) <= period:
            return avg_gain, avg_loss
            
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1w = calculate_rsi(close_1w, 14)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Weekly close > open (bullish candle) and close < open (bearish candle)
    bullish_weekly = close_1w > open_1w
    bearish_weekly = close_1w < open_1w
    bullish_weekly_aligned = align_htf_to_ltf(prices, df_1w, bullish_weekly.astype(float))
    bearish_weekly_aligned = align_htf_to_ltf(prices, df_1w, bearish_weekly.astype(float))
    
    # Volume confirmation on weekly
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(bullish_weekly_aligned[i]) or
            np.isnan(bearish_weekly_aligned[i]) or
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current weekly bar's volume for confirmation
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        vol_confirmed = vol_1w_current > 1.3 * vol_ma_1w_aligned[i]
        
        # Only look for divergence signals when flat
        if position == 0:
            # Need at least 3 periods to check for divergence
            if i >= 3:
                # Bullish divergence: price makes lower low, RSI makes higher low
                if (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                    rsi_1w_aligned[i] > rsi_1w_aligned[i-1] and rsi_1w_aligned[i-1] > rsi_1w_aligned[i-2] and
                    bullish_weekly_aligned[i] and vol_confirmed):
                    signals[i] = 0.25
                    position = 1
                    continue
                # Bearish divergence: price makes higher high, RSI makes lower high
                elif (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                      rsi_1w_aligned[i] < rsi_1w_aligned[i-1] and rsi_1w_aligned[i-1] < rsi_1w_aligned[i-2] and
                      bearish_weekly_aligned[i] and vol_confirmed):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold position if already in trade
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_1w_RSI_Divergence_Momentum_V1"
timeframe = "1d"
leverage = 1.0