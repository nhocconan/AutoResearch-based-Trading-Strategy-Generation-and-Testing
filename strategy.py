#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and volume confirmation.
Long when RSI < 30 (oversold) AND price > 4h EMA50 (uptrend bias) AND volume > 1.3x 20-period MA.
Short when RSI > 70 (overbought) AND price < 4h EMA50 (downtrend bias) AND volume > 1.3x 20-period MA.
Exit when RSI returns to neutral zone (40-60) or opposite extreme.
Uses 4h HTF for trend alignment to avoid counter-trend trades, volume for momentum confirmation.
Target: 80-120 total trades over 4 years (20-30/year) for 1h timeframe.
RSI provides mean reversion edge, 4h EMA50 filters major trend, volume confirms reversal strength.
Works in both bull and bear markets by following the higher timeframe trend while capturing short-term reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # RSI (needs 14), EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 1h volume > 1.3x 20-period MA
        vol_filter = volume[i] > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: RSI oversold (<30) AND price > 4h EMA50 (uptrend bias) AND volume filter
            if rsi_val < 30 and price > ema_val and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) AND price < 4h EMA50 (downtrend bias) AND volume filter
            elif rsi_val > 70 and price < ema_val and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: RSI returns to neutral (>=40) OR reaches overbought (>70)
                if rsi_val >= 40:
                    exit_signal = True
            elif position == -1:
                # Short exit: RSI returns to neutral (<=60) OR reaches oversold (<30)
                if rsi_val <= 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_RSI14_MeanReversion_4hEMA50_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0