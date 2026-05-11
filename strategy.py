# 1h_RSI_Bollinger_Band_Reversal_with_4hTrend
# Strategy: Mean reversion on 1h using RSI and Bollinger Bands, filtered by 4h trend
# RSI < 30 + price below lower Bollinger Band + 4h uptrend = long
# RSI > 70 + price above upper Bollinger Band + 4h downtrend = short
# Uses 4h trend (EMA50) to avoid counter-trend trades in strong moves
# Target: 15-30 trades/year per symbol to minimize fee drag
# Works in bull/bear: mean reversion works in ranges, trend filter avoids whipsaws in trends

#!/usr/bin/env python3
name = "1h_RSI_Bollinger_Band_Reversal_with_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h Bollinger Bands(20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    lower_band = sma20 - 2 * std20
    upper_band = sma20 + 2 * std20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(lower_band[i]) or np.isnan(upper_band[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold + price below lower band + 4h uptrend
            if (rsi[i] < 30 and 
                close[i] < lower_band[i] and
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + price above upper band + 4h downtrend
            elif (rsi[i] > 70 and 
                  close[i] > upper_band[i] and
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 or price > middle band
            if (rsi[i] > 50 or 
                close[i] > sma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 or price < middle band
            if (rsi[i] < 50 or 
                close[i] < sma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals