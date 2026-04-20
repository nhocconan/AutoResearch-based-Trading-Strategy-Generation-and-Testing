# 1d_RSI40_60_MeanReversion_WithTrendFilter
# Hypothesis: Mean reversion on daily timeframe with trend filter to avoid counter-trend trades
# - Uses 1-week EMA200 as trend filter: only take long positions when price above EMA200, short when below
# - Entry: RSI(14) crosses below 40 for long, crosses above 60 for short (mean reversion signals)
# - Exit: RSI returns to neutral zone (50) or opposite extreme
# - Position sizing: 0.25 for entries, 0.0 for exit
# - Designed to work in both bull and bear markets by following the higher timeframe trend
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on weekly data
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate RSI(14) on daily data
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup for EMA200
        # Skip if NaN in critical values
        if np.isnan(ema_200_1d[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current values
        price = close[i]
        rsi_val = rsi[i]
        prev_rsi = rsi[i-1]
        ema200_val = ema_200_1d[i]
        
        if position == 0:
            # Long entry: price above weekly EMA200 AND RSI crosses below 40 (oversold)
            if price > ema200_val and rsi_val < 40 and prev_rsi >= 40:
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly EMA200 AND RSI crosses above 60 (overbought)
            elif price < ema200_val and rsi_val > 60 and prev_rsi <= 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or crosses above 60 (overbought)
            if rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or crosses below 40 (oversold)
            if rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI40_60_MeanReversion_WithTrendFilter"
timeframe = "1d"
leverage = 1.0