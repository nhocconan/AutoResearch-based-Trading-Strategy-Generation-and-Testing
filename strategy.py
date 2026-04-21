#!/usr/bin/env python3
"""
6h_HTF_1d_1w_RSI_Divergence_Volume_Confirmation
Hypothesis: Use 6h primary timeframe with 1d RSI divergence (bullish/bearish) for early reversal signals in ranging markets.
Add 1w EMA50 trend filter to avoid counter-trend trades and volume confirmation (>1.6x 20-bar volume MA) to ensure momentum.
Position size 0.25 balances risk/return. Target 12-30 trades/year per symbol.
Works in bull/bear via divergence logic and EMA filter reducing whipsaw in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 14 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d RSI(14) for divergence detection ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.6 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Check 3-bar lookback for divergence
            if i >= 3:
                price_lower_low = close[i] < close[i-3] and close[i-1] < close[i-4] and close[i-2] < close[i-5]
                rsi_higher_low = rsi_1d_aligned[i] > rsi_1d_aligned[i-3] and rsi_1d_aligned[i-1] > rsi_1d_aligned[i-4] and rsi_1d_aligned[i-2] > rsi_1d_aligned[i-5]
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                price_higher_high = close[i] > close[i-3] and close[i-1] > close[i-4] and close[i-2] > close[i-5]
                rsi_lower_high = rsi_1d_aligned[i] < rsi_1d_aligned[i-3] and rsi_1d_aligned[i-1] < rsi_1d_aligned[i-4] and rsi_1d_aligned[i-2] < rsi_1d_aligned[i-5]
                
                # Long: bullish divergence + volume OK + price > 1w EMA50 (uptrend filter)
                if price_lower_low and rsi_higher_low and vol_ok and price > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish divergence + volume OK + price < 1w EMA50 (downtrend filter)
                elif price_higher_high and rsi_lower_high and vol_ok and price < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: RSI > 70 (overbought) or price < 1w EMA50 (trend change)
            if rsi_1d_aligned[i] > 70 or price < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI < 30 (oversold) or price > 1w EMA50 (trend change)
            if rsi_1d_aligned[i] < 30 or price > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_1w_RSI_Divergence_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0