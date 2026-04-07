#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h RSI Pullback with 12h EMA Trend Filter
# Hypothesis: RSI(14) pullbacks (RSI<30 in uptrend, RSI>70 in downtrend) aligned with 12h EMA(50) trend capture mean reversion within trends.
# Uses 12h EMA for trend filter (works in bull/bear) and RSI for precise entries.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

name = "6h_rsi_pullback_12h_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Align 12h EMA to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # RSI(14) on 6h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI reaches overbought or trend changes
            if rsi[i] >= 70 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI reaches oversold or trend changes
            if rsi[i] <= 30 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # RSI pullback in direction of 12h trend
            if close[i] > ema_50_aligned[i]:  # Uptrend
                if rsi[i] <= 30:  # Pullback to buy (oversold)
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                if rsi[i] >= 70:  # Pullback to sell (overbought)
                    position = -1
                    signals[i] = -0.25
    
    return signals