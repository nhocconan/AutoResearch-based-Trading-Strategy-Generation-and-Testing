# [Experiment 65707] 6h_1d_RSI_Trend_Bounce
# Hypothesis: On 6h timeframe, use 1d RSI(14) to detect mean reversion opportunities.
# Long when 1d RSI < 30 and price above 6h EMA(20); short when 1d RSI > 70 and price below 6h EMA(20).
# Uses daily RSI to avoid overtrading and capture swings in both bull and bear markets.
# Target: 15-30 trades/year per symbol with strict entry conditions.

name = "6h_1d_RSI_Trend_Bounce"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 6h EMA(20) for trend filter
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(ema20_6h[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 1d RSI oversold (<30) and price above 6h EMA20
            if rsi_1d_aligned[i] < 30 and close[i] > ema20_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1d RSI overbought (>70) and price below 6h EMA20
            elif rsi_1d_aligned[i] > 70 and close[i] < ema20_6h[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI returns to neutral (>50) or price breaks below EMA20
            if rsi_1d_aligned[i] > 50 or close[i] < ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI returns to neutral (<50) or price breaks above EMA20
            if rsi_1d_aligned[i] < 50 or close[i] > ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals