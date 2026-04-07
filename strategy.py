#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4-hour trend filter and volume confirmation
# Long when price > 4h EMA20 (uptrend), 1h RSI(14) > 50 (bullish momentum), and volume > 1.5x 1h average volume
# Short when price < 4h EMA20 (downtrend), 1h RSI(14) < 50 (bearish momentum), and volume > 1.5x 1h average volume
# Exit when trend reverses (price crosses 4h EMA20) or RSI crosses 50
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA20 for trend filter and 1h volume for confirmation
# Session filter: 08-20 UTC to reduce noise trades
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_momentum_4h_ema20_vol_v1"
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
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below 4h EMA20) or RSI < 50
            elif close[i] < ema_4h_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if in_session else 0.0
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above 4h EMA20) or RSI > 50
            elif close[i] > ema_4h_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20 if in_session else 0.0
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price above 4h EMA20 (uptrend), RSI > 50 (bullish), volume spike
            if (close[i] > ema_4h_aligned[i] and
                rsi[i] > 50 and
                volume[i] > 1.5 * volume_ma[i] and
                in_session):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price below 4h EMA20 (downtrend), RSI < 50 (bearish), volume spike
            elif (close[i] < ema_4h_aligned[i] and
                  rsi[i] < 50 and
                  volume[i] > 1.5 * volume_ma[i] and
                  in_session):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals