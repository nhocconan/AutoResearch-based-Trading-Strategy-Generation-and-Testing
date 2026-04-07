#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI pullback with 4h/1d trend filter and session filter
# Hypothesis: In strong 4h/1d trends, pullbacks to RSI(14) < 30 (long) or > 70 (short) during London/NY session (08-20 UTC) offer high-probability entries.
# Trend filter avoids counter-trend trades. Session filter reduces noise. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_rsi_pullback_4h1d_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # RSI(14) on 1h
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC (London/NY overlap)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend changes
            if rsi_values[i] > 70 or close[i] < ema_50_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend changes
            if rsi_values[i] < 30 or close[i] > ema_50_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if in_session:
                # Long: RSI oversold (<30) in uptrend (price above both EMAs)
                if (rsi_values[i] < 30 and 
                    close[i] > ema_50_4h_aligned[i] and 
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: RSI overbought (>70) in downtrend (price below both EMAs)
                elif (rsi_values[i] > 70 and 
                      close[i] < ema_50_4h_aligned[i] and 
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals