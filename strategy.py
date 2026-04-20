#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) + 1d Trend Filter + Momentum Confirmation
# - Long when price > VWAP(20) + price > 1d EMA(50) + RSI(14) > 50
# - Short when price < VWAP(20) + price < 1d EMA(50) + RSI(14) < 50
# - Exit when price crosses back through VWAP or momentum reverses
# - Uses VWAP for intraday mean reversion, 1d EMA for trend filter, RSI for momentum
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate VWAP(20) on 6h timeframe
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    volume = prices['volume'].values
    tpv = typical_price * volume
    vwap_numerator = pd.Series(tpv).rolling(window=20, min_periods=20).sum().values
    vwap_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, 0)
    
    # Calculate RSI(14) on 6h timeframe
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA/VWAP/RSI warmup
        # Skip if NaN in indicators
        if np.isnan(vwap[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long entry: price > VWAP + price > 1d EMA(50) + bullish momentum
            if price > vwap[i] and price > ema_50_1d_aligned[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short entry: price < VWAP + price < 1d EMA(50) + bearish momentum
            elif price < vwap[i] and price < ema_50_1d_aligned[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < VWAP or momentum turns bearish
            if price < vwap[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > VWAP or momentum turns bullish
            if price > vwap[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VWAP_1dEMA50_Momentum"
timeframe = "6h"
leverage = 1.0