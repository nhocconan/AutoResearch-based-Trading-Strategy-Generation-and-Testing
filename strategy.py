#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action with 12h trend filter and volume confirmation
# Long when price closes above 12h EMA50 AND 4h RSI < 30 (oversold bounce) AND 4h volume > 1.5x 20-period average
# Short when price closes below 12h EMA50 AND 4h RSI > 70 (overbought rejection) AND 4h volume > 1.5x 20-period average
# Exit when price crosses back across 12h EMA50
# Uses 12h EMA for trend filter (avoids whipsaw in ranging markets) and 4h RSI for mean-reversion entries
# Volume confirms momentum behind the move. Target: 80-150 total trades over 4 years (20-38/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h EMA50 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 4h RSI(14) for mean reversion ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # RSI calculation
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # === 4h Volume confirmation ===
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_12h_val = ema_50_12h_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma_val = vol_ma_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 4h average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit if price crosses below 12h EMA50
            if price < ema_12h_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if price crosses above 12h EMA50
            if price > ema_12h_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price above 12h EMA50 AND RSI < 30 (oversold) AND volume confirmation
            if price > ema_12h_val and rsi_val < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price below 12h EMA50 AND RSI > 70 (overbought) AND volume confirmation
            elif price < ema_12h_val and rsi_val > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12hEMA50_4hRSI_Vol1.5x"
timeframe = "4h"
leverage = 1.0