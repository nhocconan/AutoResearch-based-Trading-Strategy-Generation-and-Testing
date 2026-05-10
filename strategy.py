# 1d_KAMA_Trend_Filter_Volume_Confirm
# Hypothesis: On 1d timeframe, use KAMA to determine trend direction, enter long when KAMA trending up and RSI below 50, enter short when KAMA trending down and RSI above 50, filtered by 1w volume surge.
# This captures trend-following entries with mean-reversion timing, reducing whipsaw. Weekly filter ensures we trade with higher timeframe momentum, limiting trades to 10-25/year to avoid fee drag. Works in bull/bear via trend filter.

name = "1d_KAMA_Trend_Filter_Volume_Confirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma10_1w)
    
    # 1d data for KAMA and price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10, 2, 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs correction
    
    # Correct volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30) and RSI (14)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vol_ma10_1w_aligned[i]) or
            np.isnan(kama[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: KAMA direction (up if current > previous)
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Volume filter: current 1d volume > 1.5x 1w 10-period MA
        volume_filter = volume[i] > vol_ma10_1w_aligned[i] * 1.5
        
        if position == 0:
            # Long: KAMA rising and RSI below 50 (mean reversion within uptrend)
            if kama_rising and rsi[i] < 50 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling and RSI above 50 (mean reversion within downtrend)
            elif kama_falling and rsi[i] > 50 and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA falling or RSI overbought
            if not kama_rising or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA rising or RSI oversold
            if not kama_falling or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals