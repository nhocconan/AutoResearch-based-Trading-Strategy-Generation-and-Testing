#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h EMA filter and volume confirmation
# Long when price > 4h EMA20 + RSI(14) > 50 + volume > 1.5x average
# Short when price < 4h EMA20 + RSI(14) < 50 + volume > 1.5x average
# Uses session filter (08-20 UTC) to avoid low-liquidity hours
# Target: 60-150 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown

name = "1h_momentum_4h_ema_volume_v4"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # EMA20 calculation
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align 4h EMA20 to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # RSI(14) calculation
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price below 4h EMA20 or RSI < 40
            elif close[i] < ema20_4h_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price above 4h EMA20 or RSI > 60
            elif close[i] > ema20_4h_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and session filter
            # Long: price > 4h EMA20 + RSI > 50 + volume spike
            if (close[i] > ema20_4h_aligned[i] and 
                rsi[i] > 50 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price < 4h EMA20 + RSI < 50 + volume spike
            elif (close[i] < ema20_4h_aligned[i] and 
                  rsi[i] < 50 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals