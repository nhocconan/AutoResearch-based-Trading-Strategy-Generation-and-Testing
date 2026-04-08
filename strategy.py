#!/usr/bin/env python3
# 1h_ema_trend_4h_rsi_filter
# Hypothesis: Use 4h RSI to identify overbought/oversold conditions and 1h EMA crossover for entry timing.
# Long when price > 1h EMA20 and 4h RSI < 30 (oversold).
# Short when price < 1h EMA20 and 4h RSI > 70 (overbought).
# Exit when price crosses back over/under EMA20 or RSI returns to neutral zone (40-60).
# Designed to capture mean reversion within the trend on 1h timeframe with 4h filter.
# Target: 60-150 total trades over 4 years (~15-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_trend_4h_rsi_filter"
timeframe = "1h"
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
    
    # Get 4h data for RSI filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI(14)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h EMA20 for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_20[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20 OR RSI returns to neutral (>40)
            if (close[i] < ema_20[i]) or (rsi_4h_aligned[i] > 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20 OR RSI returns to neutral (<60)
            if (close[i] > ema_20[i]) or (rsi_4h_aligned[i] < 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price above EMA20 and 4h RSI oversold (<30)
            if (close[i] > ema_20[i]) and (rsi_4h_aligned[i] < 30):
                position = 1
                signals[i] = 0.20
            # Short: price below EMA20 and 4h RSI overbought (>70)
            elif (close[i] < ema_20[i]) and (rsi_4h_aligned[i] > 70):
                position = -1
                signals[i] = -0.20
    
    return signals