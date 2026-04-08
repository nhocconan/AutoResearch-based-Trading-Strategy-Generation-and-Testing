#!/usr/bin/env python3
"""
1d_1w_momentum_reversal_v1
Hypothesis: On daily timeframe, momentum reversals at weekly extremes with volume confirmation capture swing moves in both bull and bear markets. Weekly trend filter ensures we trade with the higher timeframe momentum, while volume filters false signals. Designed for lower frequency to avoid fee drag.
- Long: RSI(14) < 30 + price > weekly EMA(20) + volume > 1.5x 20-day average
- Short: RSI(14) > 70 + price < weekly EMA(20) + volume > 1.5x 20-day average
- Exit: RSI returns to neutral zone (40-60) or weekly trend reversal
- Position sizing: 0.25 long, -0.25 short
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_momentum_reversal_v1"
timeframe = "1d"
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
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema_20_1w
    trend_1w_down = close_1w < ema_20_1w
    
    # Forward fill weekly trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: daily volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi_values[i]) or np.isnan(trend_1w_up_aligned[i]) or 
            np.isnan(trend_1w_down_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) OR weekly trend turns down
            if (rsi_values[i] >= 40 and rsi_values[i] <= 60) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) OR weekly trend turns up
            if (rsi_values[i] >= 40 and rsi_values[i] <= 60) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: RSI oversold + price above weekly EMA + volume
            if (rsi_values[i] < 30) and trend_1w_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought + price below weekly EMA + volume
            elif (rsi_values[i] > 70) and trend_1w_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals