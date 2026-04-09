#!/usr/bin/env python3
# 6h_weekly_pivot_reversal_v1
# Hypothesis: 6h strategy using weekly pivot points for structure and 1d RSI extremes for entry timing.
# Weekly pivot levels (R2, S2) act as strong support/resistance; price rejecting these levels with
# RSI < 30 (oversold) or RSI > 70 (overbought) on the daily timeframe provides high-probability reversal entries.
# Volume confirmation (>1.3x 20-period average) filters false signals. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Works in bull/bear markets by fading extremes at key weekly levels. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_d = df_1d['close'].values
    
    # 1d RSI(14)
    delta = pd.Series(close_d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    
    # Align 1d RSI to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Weekly pivot points from 1d data (using prior week's OHLC)
    # We'll approximate weekly OHLC by resampling 1d data weekly
    # But to avoid look-ahead, we use the completed prior week's data
    # For simplicity, we'll use the prior 5 trading days' OHLC as weekly proxy
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly high, low, close from prior 5 days (updated daily)
    # We need to shift by 1 to use only completed weekly data
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1, S1
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    # Weekly R2, S2 (stronger levels)
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 60 (overbought) or price breaks below weekly pivot
            if rsi_1d_aligned[i] > 60 or close[i] < weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 (oversold) or price breaks above weekly pivot
            if rsi_1d_aligned[i] < 40 or close[i] > weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price at or above weekly S2 AND RSI < 30 (oversold)
                if close[i] >= weekly_s2_aligned[i] and rsi_1d_aligned[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price at or below weekly R2 AND RSI > 70 (overbought)
                elif close[i] <= weekly_r2_aligned[i] and rsi_1d_aligned[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals