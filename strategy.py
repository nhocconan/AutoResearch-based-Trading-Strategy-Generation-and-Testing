# 1d_RSI_50_Crossover_1wMA100_Trend_Volume
# Hypothesis: Daily RSI crossing above/below 50 with weekly MA100 trend filter and volume confirmation.
# RSI(14) > 50 indicates bullish momentum, < 50 bearish. Weekly MA100 filters for long-term trend direction.
# Volume > 1.5x 20-day EMA confirms participation. Works in both bull/bear by following weekly trend.
# Target: 20-40 trades/year on daily timeframe.
name = "1d_RSI_50_Crossover_1wMA100_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first value as NaN to align with price array
    rsi = np.concatenate([[np.nan], rsi])
    
    # Weekly MA100 trend filter
    ma_100_1w = pd.Series(df_1w['close'].values).rolling(window=100, min_periods=100).mean().values
    ma_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ma_100_1w)
    
    # Volume confirmation: volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for RSI and weekly MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(rsi[i]) or np.isnan(ma_100_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI > 50 (bullish momentum) + price above weekly MA100 + volume confirmation
            if rsi[i] > 50 and price > ma_100_1w_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 (bearish momentum) + price below weekly MA100 + volume confirmation
            elif rsi[i] < 50 and price < ma_100_1w_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses below 50 or price below weekly MA100
            if rsi[i] < 50 or price < ma_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 50 or price above weekly MA100
            if rsi[i] > 50 or price > ma_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals