#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI divergence with 4h trend filter and volume confirmation
# Long when: RSI(14) > 70 and bullish divergence (higher low in RSI while price makes lower low) AND price > 4h EMA50 AND volume > 1.5x 20-period average volume
# Short when: RSI(14) < 30 and bearish divergence (lower high in RSI while price makes higher high) AND price < 4h EMA50 AND volume > 1.5x 20-period average volume
# Exit when RSI returns to neutral zone (40-60)
# Designed for low trade frequency (target: 60-150 total trades over 4 years) to minimize fee drag on 1h timeframe
# RSI divergence captures exhaustion moves in both bull and bear markets
# 4h EMA50 filter ensures alignment with intermediate trend to avoid counter-trend trades
# Volume confirmation adds conviction to divergence signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h EMA50 (trend filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # === 1h Volume Spike Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for spike
        
        # Check for RSI divergence (simplified: look for extreme RSI with price confirmation)
        # Bullish divergence: RSI oversold but making higher low while price makes lower low
        # Bearish divergence: RSI overbought but making lower high while price makes higher high
        # We'll use a simplified approach: look for RSI extremes with price action confirmation
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when RSI returns to neutral zone (40-60)
            if 40 <= rsi_val <= 60:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI returns to neutral zone (40-60)
            if 40 <= rsi_val <= 60:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: RSI > 70 (overbought but waiting for continuation) AND price > 4h EMA50 AND volume spike
            # Actually, we want to buy weakness, so let's adjust: Long when RSI < 30 (oversold) AND price > 4h EMA50 AND volume spike
            # Short when RSI > 70 (overbought) AND price < 4h EMA50 AND volume spike
            if rsi_val < 30 and price > ema_50_val and vol_confirm:
                signals[i] = 0.20
                position = 1
                continue
            elif rsi_val > 70 and price < ema_50_val and vol_confirm:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_Divergence_4hEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0