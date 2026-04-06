#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h RSI filter and 1d volatility filter
# Long when RSI(14) < 30, price > 4h EMA(200) (bullish trend), and 1d volatility < median (low volatility regime)
# Short when RSI(14) > 70, price < 4h EMA(200) (bearish trend), and 1d volatility < median
# Exit when RSI returns to 50 or opposite signal occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20
# Uses 4h EMA(200) for trend filter and 1d ATR ratio for volatility regime filter
# Target: 60-150 total trades over 4 years (15-38/year)
# Session filter: 08-20 UTC to avoid low-volume periods

name = "1h_rsi14_4h_ema200_1d_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA(200) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=200, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio (current ATR / 50-period median ATR) for volatility regime
    atr_series = pd.Series(atr_1d)
    atr_median = atr_series.rolling(window=50, min_periods=50).median().values
    atr_ratio = atr_1d / atr_median
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # RSI(14) on 1h
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
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volatility regime: only trade in low volatility (ATR ratio < 1.2)
        low_vol_regime = atr_ratio_aligned[i] < 1.2
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI returns to 50 or bearish signal
            elif rsi[i] >= 50 or (rsi[i] > 70 and close[i] < ema_4h_aligned[i]):
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
            # Exit: RSI returns to 50 or bullish signal
            elif rsi[i] <= 50 or (rsi[i] < 30 and close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with RSI extremes, trend alignment, and low volatility
            # Long: RSI < 30 (oversold), price above EMA (bullish trend), low volatility
            if (rsi[i] < 30 and
                close[i] > ema_4h_aligned[i] and
                low_vol_regime):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI > 70 (overbought), price below EMA (bearish trend), low volatility
            elif (rsi[i] > 70 and
                  close[i] < ema_4h_aligned[i] and
                  low_vol_regime):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals