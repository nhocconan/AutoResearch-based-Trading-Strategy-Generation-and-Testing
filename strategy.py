#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Uses 4h EMA50 for trend direction (trend filter) and 1h RSI(14) for mean reversion entries.
# Long when RSI < 30 and price > 4h EMA50; short when RSI > 70 and price < 4h EMA50.
# Volume confirmation requires current volume > 1.5x 20-period average.
# Includes session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.

name = "1h_RSI_MeanRev_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # === 4h EMA50 for trend direction ===
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # === 1h RSI(14) for mean reversion ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute session hours
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after RSI warmup
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) and price above 4h EMA50 (uptrend filter)
            if rsi[i] < 30 and prices['close'].iloc[i] > ema_50_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) and price below 4h EMA50 (downtrend filter)
            elif rsi[i] > 70 and prices['close'].iloc[i] < ema_50_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or trend reversal
            if rsi[i] > 50 or prices['close'].iloc[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or trend reversal
            if rsi[i] < 50 or prices['close'].iloc[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals