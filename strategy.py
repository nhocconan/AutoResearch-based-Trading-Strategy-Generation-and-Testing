#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI with 12h trend filter and volume confirmation.
# Uses RSI(14) overbought/oversold levels but only trades in direction of 12h EMA50 trend.
# This avoids counter-trend trades in strong trends while capturing mean reversion in ranges.
# Volume confirmation ensures institutional participation. Designed for 6-12 trades per year.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "6h_RSI_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h EMA50 for trend direction ===
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === 6h RSI(14) ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        rsi_val = rsi[i]
        ema_val = ema_50_aligned[i]
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) in uptrend (price > EMA50) with volume
            if rsi_val < 30 and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in downtrend (price < EMA50) with volume
            elif rsi_val > 70 and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend change (price < EMA50)
            if rsi_val > 70 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend change (price > EMA50)
            if rsi_val < 30 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals