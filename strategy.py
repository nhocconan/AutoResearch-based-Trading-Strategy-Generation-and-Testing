#!/usr/bin/env python3
# 6h_rsi_mean_reversion_trend_filter_v1
# Hypothesis: In trending markets, RSI mean reversion (RSI<30 long, RSI>70 short) provides high-probability entries when aligned with higher timeframe trend (12h EMA50). Volume confirmation filters out low-quality signals. Works in both bull and bear markets by following the trend while exploiting short-term overextensions.
# Target: 20-30 trades/year via strict RSI extremes + trend alignment + volume filter.

name = "6h_rsi_mean_reversion_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 20-period average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for higher timeframe trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(50, 20)  # Need enough data for RSI and volume average
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Get aligned 12h EMA50 for trend filter
        ema50_12h_val = align_htf_to_ltf(prices, df_12h, ema50_12h)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_12h_val) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if RSI crosses above 50 (mean reversion complete) OR trend changes
            if rsi[i] > 50 or ema50_12h_val <= close[i]:  # Trend filter: price below 12h EMA50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if RSI crosses below 50 (mean reversion complete) OR trend changes
            if rsi[i] < 50 or ema50_12h_val >= close[i]:  # Trend filter: price above 12h EMA50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long setup: RSI oversold (<30) with volume filter, aligned with uptrend
            if (rsi[i] < 30 and 
                volume_filter and 
                ema50_12h_val < close[i]):  # Price above 12h EMA50 = uptrend
                position = 1
                signals[i] = 0.25
            # Short setup: RSI overbought (>70) with volume filter, aligned with downtrend
            elif (rsi[i] > 70 and 
                  volume_filter and 
                  ema50_12h_val > close[i]):  # Price below 12h EMA50 = downtrend
                position = -1
                signals[i] = -0.25
    
    return signals