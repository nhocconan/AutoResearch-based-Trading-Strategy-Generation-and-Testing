#!/usr/bin/env python3
"""
4h_RSI_40_60_MeanReversion_1dTrend_Volume
Hypothesis: On 4h timeframe, take mean reversion trades when RSI(14) reaches extreme levels (40 for long, 60 for short) only when aligned with daily trend (price vs EMA50) and confirmed by volume surge. This combines mean reversion in ranging markets with trend filter to avoid counter-trend trades in strong trends, working in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions with trend alignment and volume surge
        # Long: RSI < 40 (oversold) + daily uptrend + volume surge
        long_entry = (rsi_values[i] < 40 and 
                     trend_up and 
                     volume_surge[i])
        
        # Short: RSI > 60 (overbought) + daily downtrend + volume surge
        short_entry = (rsi_values[i] > 60 and 
                      trend_down and 
                      volume_surge[i])
        
        # Exit when RSI returns to neutral zone (40-60) with volume surge
        long_exit = (rsi_values[i] > 50 and volume_surge[i])
        short_exit = (rsi_values[i] < 50 and volume_surge[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_40_60_MeanReversion_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0