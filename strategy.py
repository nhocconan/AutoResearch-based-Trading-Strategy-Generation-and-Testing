#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI + 4h EMA200 Trend Filter + Session Filter (08-20 UTC)
# - RSI(14) on 1h for momentum reversal signals
# - Long when RSI < 30 and price > 4h EMA200 (oversold in uptrend)
# - Short when RSI > 70 and price < 4h EMA200 (overbought in downtrend)
# - EMA200 filters for long-term trend alignment to avoid counter-trend trades
# - Session filter (08-20 UTC) reduces noise during low-volume periods
# - Target: 15-37 trades per year per symbol (60-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 4h data for EMA200 calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA200 on 4h timeframe
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMA200 to 1h timeframe
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate RSI on 1h timeframe
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral RSI when no loss
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if NaN in indicators
        if np.isnan(rsi[i]) or np.isnan(ema_200_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema200 = ema_200_4h_aligned[i]
        
        if position == 0:
            # Long entry: RSI oversold (< 30) + price above 4h EMA200
            if rsi_val < 30 and price > ema200:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI overbought (> 70) + price below 4h EMA200
            elif rsi_val > 70 and price < ema200:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI rises above 50 or price falls below EMA200
            if rsi_val > 50 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI falls below 50 or price rises above EMA200
            if rsi_val < 50 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_4hEMA200_TrendFilter_Session"
timeframe = "1h"
leverage = 1.0