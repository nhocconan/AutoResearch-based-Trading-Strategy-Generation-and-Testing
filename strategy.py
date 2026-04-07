#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4-hour trend filter and daily volume confirmation
# Long when 1h RSI(14) crosses above 50, 4h close > 4h EMA50 (uptrend), and 1h volume > 1.5x daily average volume
# Short when 1h RSI(14) crosses below 50, 4h close < 4h EMA50 (downtrend), and 1h volume > 1.5x daily average volume
# Exit when RSI crosses back to 50 or opposite signal occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA50 for trend filter and daily volume average for confirmation
# Target: 100-180 total trades over 4 years (25-45/year)

name = "1h_momentum_4h_ema50_daily_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # RSI(14) for momentum
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
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses below 50 or trend reverses
            elif rsi[i] < 50 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses above 50 or trend reverses
            elif rsi[i] > 50 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: RSI crosses above 50, price above EMA50 (uptrend), volume spike
            if (rsi[i] > 50 and rsi[i-1] <= 50 and
                close[i] > ema_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI crosses below 50, price below EMA50 (downtrend), volume spike
            elif (rsi[i] < 50 and rsi[i-1] >= 50 and
                  close[i] < ema_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals