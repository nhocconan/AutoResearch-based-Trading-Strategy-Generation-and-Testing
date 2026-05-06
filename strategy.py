#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h EMA trend filter and 6h RSI mean reversion
# - Uses 12h EMA50 to establish trend direction (higher timeframe filter)
# - Uses 6h RSI(14) for mean reversion entries during pullbacks in trend
# - Enters long when RSI < 30 and price > 12h EMA50 (oversold in uptrend)
# - Enters short when RSI > 70 and price < 12h EMA50 (overbought in downtrend)
# - Uses volume confirmation (volume > 1.5x 20-period average) to filter false signals
# - Exits when RSI returns to neutral zone (40-60) or trend changes
# - Designed to capture mean reversion within established trends, working in both bull and bear markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_12hEMA50_RSI_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price above 12h EMA50 (uptrend) + volume confirmation
            if rsi[i] < 30 and close[i] > ema_50_6h[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price below 12h EMA50 (downtrend) + volume confirmation
            elif rsi[i] > 70 and close[i] < ema_50_6h[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>=40) OR trend turns bearish (price < EMA50)
            if rsi[i] >= 40 or close[i] < ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (<=60) OR trend turns bullish (price > EMA50)
            if rsi[i] <= 60 or close[i] > ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals