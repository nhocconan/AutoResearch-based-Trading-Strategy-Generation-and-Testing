#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend Filter + Volume Spike
Long: Price pulls back to 4h EMA21 (pullback < 1.5%) + RSI(14) < 30 + volume > 1.5x 1h volume SMA(20)
Short: Price rallies to 4h EMA21 (pullback < 1.5%) + RSI(14) > 70 + volume > 1.5x 1h volume SMA(20)
Exit: Opposite RSI condition or price moves >1% away from 4h EMA21
Uses 4h EMA for trend direction and 1h for precise entry timing with mean reversion.
Target: 60-150 total trades over 4 years (15-37/year)
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
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(21)
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.fillna(50).values  # neutral when undefined
    
    # Calculate 1h volume SMA(20)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 21)  # need RSI and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi_14[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        rsi_val = rsi_14[i]
        ema_21_val = ema_21_4h_aligned[i]
        
        # Calculate pullback percentage to 4h EMA21
        pullback_pct = abs(price - ema_21_val) / ema_21_val * 100
        
        if position == 0:
            # Long: Pullback to 4h EMA21 + RSI oversold + volume spike
            if pullback_pct < 1.5 and price <= ema_21_val and rsi_val < 30 and vol > 1.5 * vol_sma_val:
                signals[i] = 0.20
                position = 1
            # Short: Rally to 4h EMA21 + RSI overbought + volume spike
            elif pullback_pct < 1.5 and price >= ema_21_val and rsi_val > 70 and vol > 1.5 * vol_sma_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 or price moves >1% above EMA21
            if rsi_val > 50 or price > ema_21_val * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI < 50 or price moves >1% below EMA21
            if rsi_val < 50 or price < ema_21_val * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Pullback_4hEMA21_VolumeSpike"
timeframe = "1h"
leverage = 1.0