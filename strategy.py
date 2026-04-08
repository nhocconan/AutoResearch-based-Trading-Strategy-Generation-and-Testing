#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d RSI mean reversion and volume filter
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI > 70 or < 30 on daily timeframe indicates overbought/oversold conditions for mean reversion.
# Volume > 1.3x 20-period average confirms momentum.
# Works in both bull and bear markets by combining trend following with mean reversion.
# Target: 20-50 trades/year by requiring KAMA trend + RSI extreme + volume confirmation.
name = "4h_kama_trend_1d_rsi_meanrev_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on 1d data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate KAMA on 4h data
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix shapes
    change = np.concatenate([[np.nan]*10, change[10:]])
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    
    er = change / (volatility + 1e-10)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 4h bar
        rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)[i]
        
        # Trend filter: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI mean reversion: extreme levels
        rsi_overbought = rsi_aligned > 70
        rsi_oversold = rsi_aligned < 30
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI becomes overbought
            if not price_above_kama or rsi_overbought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI becomes oversold
            if not price_below_kama or rsi_oversold:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_confirm:
                # Long: price above KAMA AND RSI oversold (trend + mean reversion)
                if price_above_kama and rsi_oversold:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA AND RSI overbought (trend + mean reversion)
                elif price_below_kama and rsi_overbought:
                    position = -1
                    signals[i] = -0.25
    
    return signals