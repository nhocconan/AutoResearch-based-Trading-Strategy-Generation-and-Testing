#!/usr/bin/env python3
"""
4h_RSI_Trend_Squeeze_v1
Strategy: 4h RSI with trend filter and volatility squeeze filter.
Long: RSI(14) crosses above 50 in uptrend during low volatility (BBW < 20th percentile).
Short: RSI(14) crosses below 50 in downtrend during low volatility (BBW < 20th percentile).
Exit: RSI crosses back below/above 50 or volatility expands (BBW > 50th percentile).
Designed for 4h timeframe: ~20-40 trades/year per symbol (80-160 total over 4 years).
Works in bull/bear via trend filter and volatility regime filter.
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
    
    # Get daily data for trend filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Daily Bollinger Bands for volatility regime (20, 2)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # Percentiles for volatility regime (use expanding window to avoid look-ahead)
    bb_width_series = pd.Series(bb_width_1d)
    bb_width_pct_20 = bb_width_series.expanding(min_periods=50).quantile(0.20).values
    bb_width_pct_50 = bb_width_series.expanding(min_periods=50).quantile(0.50).values
    
    bb_width_pct_20_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct_20)
    bb_width_pct_50_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct_50)
    
    # 4h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_100_aligned[i]) or np.isnan(bb_width_pct_20_aligned[i]) or 
            np.isnan(bb_width_pct_50_aligned[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition
        uptrend = close[i] > ema_100_aligned[i]
        downtrend = close[i] < ema_100_aligned[i]
        
        # Volatility squeeze condition (low volatility)
        low_volatility = bb_width_1d[i] < bb_width_pct_20_aligned[i]
        high_volatility = bb_width_1d[i] > bb_width_pct_50_aligned[i]
        
        # RSI conditions
        rsi_cross_up = rsi_values[i] > 50 and rsi_values[i-1] <= 50
        rsi_cross_down = rsi_values[i] < 50 and rsi_values[i-1] >= 50
        
        if position == 0:
            # Long: uptrend + low volatility + RSI cross up
            if uptrend and low_volatility and rsi_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + low volatility + RSI cross down
            elif downtrend and low_volatility and rsi_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volatility expansion, or RSI cross down
            if not uptrend or high_volatility or rsi_cross_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volatility expansion, or RSI cross up
            if not downtrend or high_volatility or rsi_cross_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Trend_Squeeze_v1"
timeframe = "4h"
leverage = 1.0