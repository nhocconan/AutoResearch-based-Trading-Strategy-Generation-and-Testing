# 2025-01-01-1d-01
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week EMA10 trend filter with RSI(14) mean reversion
# - Long when RSI < 30 (oversold) and price > weekly EMA10 (uptrend filter)
# - Short when RSI > 70 (overbought) and price < weekly EMA10 (downtrend filter)
# - Exit when RSI crosses back to neutral (40-60 range)
# - Designed to capture mean-reversion bounces in trending markets
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_RSI14_MeanReversion_1wEMA10"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1w data for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate RSI(14) on daily timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price > weekly EMA10 (uptrend)
            if rsi[i] < 30 and close[i] > ema_10_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price < weekly EMA10 (downtrend)
            elif rsi[i] > 70 and close[i] < ema_10_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses back to neutral (>= 40)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses back to neutral (<= 60)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals