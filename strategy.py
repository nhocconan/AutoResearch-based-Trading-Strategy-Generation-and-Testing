#!/usr/bin/env python3
"""
4h_1d_rsi_mean_reversion_v1
Hypothesis: RSI mean reversion on 4h with 1d trend filter. 
- Long when RSI < 30 and 1d close > EMA50 (bullish bias in uptrend)
- Short when RSI > 70 and 1d close < EMA50 (bearish bias in downtrend)
- Volume filter: 4h volume > 1.5x 20-period average to confirm momentum
- Exit: RSI crosses back through 50
- Position sizing: 0.25 long, -0.25 short
- Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_mean_reversion_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema_50_1d
    trend_1d_down = close_1d < ema_50_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # 4h RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50
            if rsi_values[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50
            if rsi_values[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: RSI < 30 + 1d uptrend + volume
            if (rsi_values[i] < 30) and trend_1d_up_aligned[i] and volume_filter[i]:
                # Confirm RSI was above 30 previous bar
                if i > start_idx and rsi_values[i-1] >= 30:
                    position = 1
                    signals[i] = 0.25
            # Short entry: RSI > 70 + 1d downtrend + volume
            elif (rsi_values[i] > 70) and trend_1d_down_aligned[i] and volume_filter[i]:
                # Confirm RSI was below 70 previous bar
                if i > start_idx and rsi_values[i-1] <= 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals