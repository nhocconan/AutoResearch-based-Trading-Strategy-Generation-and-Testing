#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Keltner channel (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Keltner channel on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR on 1w
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=20, min_periods=20).mean().values
    
    # EMA on 1w
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20).mean().values
    
    # Keltner bands on 1w
    upper_1w = ema_1w + 2 * atr_1w
    lower_1w = ema_1w - 2 * atr_1w
    
    # Align to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # RSI on daily close
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(ema_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion: price touches Keltner bands with RSI confirmation
        touch_upper = close[i] >= upper_aligned[i]
        touch_lower = close[i] <= lower_aligned[i]
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        # Long: touches lower band in oversold conditions
        long_signal = touch_lower and rsi_oversold
        # Short: touches upper band in overbought conditions
        short_signal = touch_upper and rsi_overbought
        
        # Exit when price returns to EMA
        exit_long = close[i] < ema_aligned[i]
        exit_short = close[i] > ema_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals