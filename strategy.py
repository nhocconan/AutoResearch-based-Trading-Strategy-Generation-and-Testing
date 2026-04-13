#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
    # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.5x avg
    # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.5x avg
    # Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit) or volume dry-up
    # Using 12h primary for low trade frequency, Williams %R for mean reversion in ranges,
    # 1d EMA for trend filter to avoid counter-trend trades, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, ((highest_high - close) / denominator) * -100, -50)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA50 = uptrend bias, price < EMA50 = downtrend bias
        uptrend_bias = close[i] > ema_1d_aligned[i]
        downtrend_bias = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Williams %R extremes + trend bias + volume confirmation
        long_entry = (williams_r[i] < -80) and uptrend_bias and vol_confirm
        short_entry = (williams_r[i] > -20) and downtrend_bias and vol_confirm
        
        # Exit logic: Williams %R crosses mid-level (-50) or volume dry-up
        long_exit = (williams_r[i] > -50) or not vol_confirm
        short_exit = (williams_r[i] < -50) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williams_r_ema_volume_v1"
timeframe = "12h"
leverage = 1.0