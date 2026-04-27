#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI mean reversion with weekly trend filter and volume confirmation.
# Long when weekly trend is up (price > weekly SMA50), daily RSI < 30, and volume > 1.5x average.
# Short when weekly trend is down (price < weekly SMA50), daily RSI > 70, and volume > 1.5x average.
# Uses weekly trend to avoid counter-trend trades in strong trends, and RSI extremes for mean reversion.
# Volume filter ensures trades occur during periods of heightened interest.
# Designed for low trade frequency (~10-20 trades/year) to minimize fee drag.

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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly SMA50 for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: weekly uptrend, RSI oversold, volume spike
        if (close[i] > sma50_1w_aligned[i] and 
            rsi_values[i] < 30 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: weekly downtrend, RSI overbought, volume spike
        elif (close[i] < sma50_1w_aligned[i] and 
              rsi_values[i] > 70 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_RSI_MeanReversion_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0