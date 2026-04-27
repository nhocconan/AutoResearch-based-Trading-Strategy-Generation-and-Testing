#!/usr/bin/env python3
"""
4h_Momentum_Confluence_BasedOn4hAnd12h_Energy
Hypothesis: Combines 4h RSI momentum with 12h EMA trend and volume confirmation to capture strong moves while avoiding whipsaws. 
In bull markets: rides momentum with trend filter. In bear markets: avoids counter-trend trades via 12h EMA filter. 
Volume surge confirms institutional interest. Discrete sizing (0.25) limits risk. Target: ~100 trades over 4 years.
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
    
    # 4h RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema12 = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema12_aligned = align_htf_to_ltf(prices, df_12h, ema12)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm)
    
    # Volatility filter: ATR ratio to avoid extreme volatility
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr < (atr_ma * 2.0)  # Avoid periods of extreme volatility
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # Position size: 25% of capital
    
    # Warmup: need RSI (14), EMA12 (21), volume avg (20), ATR (14+50)
    start_idx = max(14, 21, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema12_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema12_val = ema12_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        vol_filter = volatility_filter[i]
        close_val = close[i]
        
        if position == 0:
            # Long: RSI > 55 (momentum) + price > 12h EMA (trend) + volume + volatility filter
            if rsi_val > 55 and close_val > ema12_val and vol_conf and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI < 45 (momentum) + price < 12h EMA (trend) + volume + volatility filter
            elif rsi_val < 45 and close_val < ema12_val and vol_conf and vol_filter:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: RSI < 40 (loss of momentum) or price < 12h EMA (trend change)
            if rsi_val < 40 or close_val < ema12_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: RSI > 60 (loss of momentum) or price > 12h EMA (trend change)
            if rsi_val > 60 or close_val > ema12_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Momentum_Confluence_BasedOn4hAnd12h_Energy"
timeframe = "4h"
leverage = 1.0