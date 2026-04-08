#!/usr/bin/env python3
# 1d_rsi_ema_pullback_1w_trend
# Hypothesis: Buy pullbacks to EMA on daily timeframe during weekly uptrends and sell rallies to EMA during weekly downtrends.
# Uses RSI(14) to identify oversold/overbought conditions for entry, EMA(50) for trend and dynamic support/resistance,
# and weekly EMA(20) for higher timeframe trend filter. Designed to capture swing moves in both bull and bear markets.
# Target: 15-25 trades/year (~60-100 total over 4 years) with controlled risk via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi_ema_pullback_1w_trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily EMA(50) for trend and pullback zone
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14) for overbought/oversold conditions
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_50[i]) or np.isnan(rsi_values[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA(50) or RSI becomes overbought
            if close[i] < ema_50[i] or rsi_values[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA(50) or RSI becomes oversold
            if close[i] > ema_50[i] or rsi_values[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Weekly trend filter: only take longs in weekly uptrend, shorts in weekly downtrend
            weekly_uptrend = close[i] > ema_20_1w_aligned[i]
            weekly_downtrend = close[i] < ema_20_1w_aligned[i]
            
            # Long entry: price near EMA(50) support during weekly uptrend with RSI oversold
            if weekly_uptrend and abs(close[i] - ema_50[i]) / ema_50[i] < 0.02 and rsi_values[i] < 30:
                position = 1
                signals[i] = 0.25
            # Short entry: price near EMA(50) resistance during weekly downtrend with RSI overbought
            elif weekly_downtrend and abs(close[i] - ema_50[i]) / ema_50[i] < 0.02 and rsi_values[i] > 70:
                position = -1
                signals[i] = -0.25
    
    return signals