#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volatility regime
    # Long when: RSI(14) < 30 AND price > 4h EMA(50) AND 1d ATR ratio < 0.8 (low vol regime)
    # Short when: RSI(14) > 70 AND price < 4h EMA(50) AND 1d ATR ratio < 0.8
    # Exit when: RSI crosses 50 OR volatility regime shifts (ATR ratio > 1.2)
    # Uses discrete sizing (0.20) targeting 60-150 trades over 4 years.
    # Works in bull/bear via 4h EMA trend filter and 1d volatility regime avoiding chop.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h ATR(14) for comparison
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = 0
    tr2_h[0] = 0
    tr3_h[0] = 0
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_1h = pd.Series(tr_h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: 1h ATR / 1d ATR (measures short-term volatility relative to daily)
    atr_ratio = atr_1h / (atr_1d + 1e-10)  # Avoid division by zero
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(atr_ratio[i-1])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volatility regime filter: low volatility environment (ATR ratio < 0.8)
        vol_regime_ok = atr_ratio[i] < 0.8
        
        # Trend filter: price relative to 4h EMA(50)
        price_above_ema = close[i] > ema_4h_aligned[i]
        price_below_ema = close[i] < ema_4h_aligned[i]
        
        # Mean reversion signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_exit = abs(rsi[i] - 50) < 2  # Exit near RSI 50
        
        # Entry conditions
        long_entry = rsi_oversold and price_above_ema and vol_regime_ok and in_session and position != 1
        short_entry = rsi_overbought and price_below_ema and vol_regime_ok and in_session and position != -1
        
        # Exit conditions: RSI mean reversion OR volatility regime shift
        exit_long = rsi_exit or (atr_ratio[i] > 1.2) or not in_session
        exit_short = rsi_exit or (atr_ratio[i] > 1.2) or not in_session
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_rsi_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0