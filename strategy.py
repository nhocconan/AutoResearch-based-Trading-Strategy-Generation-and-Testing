#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI mean reversion with 4-hour trend filter and session filter (08-20 UTC)
# Long when RSI(14) < 30 + 4-hour EMA(50) up + session active
# Short when RSI(14) > 70 + 4-hour EMA(50) down + session active
# Exit when RSI crosses 50 in opposite direction
# Stoploss at 2.5 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4-hour EMA for trend direction and session filter to avoid low-volatility periods
# Target: 100-180 total trades over 4 years (25-45/year)

name = "1h_rsi14_4h_ema50_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) on 1-hour
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_active = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or not session_active[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses above 50
            elif rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses below 50
            elif rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + 4-hour EMA trend + session
            # RSI oversold/overbought
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            
            # 4-hour EMA trend: slope approximation
            ema_up = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] if i > 0 else False
            ema_down = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] if i > 0 else False
            
            # Long: RSI oversold + 4-hour EMA up + session
            if rsi_oversold and ema_up and session_active[i]:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI overbought + 4-hour EMA down + session
            elif rsi_overbought and ema_down and session_active[i]:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals