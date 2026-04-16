#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# Long when RSI(14) < 30 AND price > 4h EMA50 AND volume > 1.3x 20-period average volume
# Short when RSI(14) > 70 AND price < 4h EMA50 AND volume > 1.3x 20-period average volume
# Exit when RSI returns to neutral (40-60 range) or opposing signal appears
# Designed for low trade frequency (target: 60-150 total trades over 4 years) to minimize fee drag on 1h timeframe
# RSI mean reversion works in ranging markets, EMA50 filter avoids counter-trend trades in trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === RSI(14) calculation ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral RSI when insufficient data
    
    # === 4h EMA50 (trend filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_ma_val = vol_ma_20[i]
        vol_confirm = volume[i] > vol_ma_val * 1.3  # 1.3x average volume for confirmation
        
        # === EXIT CONDITIONS ===
        if position == 1:  # Long position
            # Exit when RSI returns to neutral (40-60) or opposing signal
            if rsi_val >= 40 or (rsi_val > 70 and price < ema_50_val and vol_confirm):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit when RSI returns to neutral (40-60) or opposing signal
            if rsi_val <= 60 or (rsi_val < 30 and price > ema_50_val and vol_confirm):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) AND volume confirmation
            if rsi_val < 30 and price > ema_50_val and vol_confirm:
                signals[i] = 0.20
                position = 1
                continue
            # Short when: RSI > 70 (overbought) AND price < 4h EMA50 (downtrend) AND volume confirmation
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

name = "1h_RSI14_4hEMA50_VolumeConfirm_MeanRev"
timeframe = "1h"
leverage = 1.0