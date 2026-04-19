#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h/1d regime filter and volume confirmation.
# Long when: 4h ADX > 25 (trending) AND 1h RSI < 40 (oversold) AND volume > 1.5x 20-period average
# Short when: 4h ADX > 25 (trending) AND 1h RSI > 60 (overbought) AND volume > 1.5x 20-period average
# Exit when RSI returns to neutral range (40-60)
# Uses 4h for trend strength filter, 1h for mean reversion entries, volume for confirmation.
# Session filter: 08-20 UTC to avoid low-volume periods.
# Target: 15-37 trades/year per symbol (60-150 total over 4 years).

name = "1h_ADX_RSI_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX (trend strength filter)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate ADX on 4h data
    # True Range
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = np.abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((df_4h['high'] - df_4h['high'].shift(1)) > (df_4h['low'].shift(1) - df_4h['low']),
                       np.maximum(df_4h['high'] - df_4h['high'].shift(1), 0), 0)
    dm_minus = np.where((df_4h['low'].shift(1) - df_4h['low']) > (df_4h['high'] - df_4h['high'].shift(1)),
                        np.maximum(df_4h['low'].shift(1) - df_4h['low'], 0), 0)
    
    # Smooth DM
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Get 1d data for additional regime filter (optional trend bias)
    df_1d = get_htf_data(prices, '1d')
    # Simple trend: price vs 50 EMA on 1d
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h indicators
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 50)  # ADX, volume MA, and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        rsi_val = rsi[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Trend strength filter: need trending market (ADX > 25)
        is_trending = adx_val > 25
        
        if position == 0:
            # Long: trending + oversold RSI + volume spike
            if is_trending and rsi_val < 40 and vol_ratio > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: trending + overbought RSI + volume spike
            elif is_trending and rsi_val > 60 and vol_ratio > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or overbought
            if rsi_val >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral or oversold
            if rsi_val <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals