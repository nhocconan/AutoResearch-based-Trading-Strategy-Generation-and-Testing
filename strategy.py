#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour trading with 4h trend filter and 1d volatility filter
# Long when: price > 4h EMA200 (trend), price > 1d VWAP (institutional bias), and hourly RSI < 30 (oversold bounce)
# Short when: price < 4h EMA200 (trend), price < 1d VWAP (institutional bias), and hourly RSI > 70 (overbought rejection)
# Exit when RSI crosses back to neutral (50) or trend filter fails
# Uses 4h for trend direction, 1d for institutional bias, 1h for precise entry timing
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Session filter: 08-20 UTC to avoid low-liquidity hours

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA200 (trend filter)
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 1d VWAP (institutional bias)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_1d = (np.cumsum(typical_price_1d * df_1d['volume'].values) / np.cumsum(df_1d['volume'].values))
    
    # Calculate 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 1h timeframe
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # for RSI and EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(rsi[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: above 4h EMA200, above 1d VWAP, and RSI oversold
            if (price > ema_200_4h_aligned[i] and 
                price > vwap_1d_aligned[i] and 
                rsi[i] < 30):
                position = 1
                signals[i] = position_size
            # Short setup: below 4h EMA200, below 1d VWAP, and RSI overbought
            elif (price < ema_200_4h_aligned[i] and 
                  price < vwap_1d_aligned[i] and 
                  rsi[i] > 70):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 or trend fails
            if rsi[i] > 50 or price < ema_200_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50 or trend fails
            if rsi[i] < 50 or price > ema_200_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hEMA200_1dVWAP_RSI"
timeframe = "1h"
leverage = 1.0