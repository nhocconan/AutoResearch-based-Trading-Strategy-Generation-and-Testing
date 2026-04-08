#!/usr/bin/env python3
# 1h_4d_volume_confirmation_v1
# Hypothesis: Trade mean reversion in 1h timeframe with 4h trend filter and volume confirmation.
# In bullish 4h regime (price > 4h EMA50): long when 1h RSI < 30 and volume > 1.5x average.
# In bearish 4h regime (price < 4h EMA50): short when 1h RSI > 70 and volume > 1.5x average.
# Uses RSI for mean-reversion entries and volume to confirm momentum.
# Designed for 1h timeframe targeting 15-37 trades/year (60-150 total over 4 years).
# Works in bull/bear markets by aligning with 4h trend while exploiting 1h overextensions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_volume_confirmation_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI(14) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure RSI and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_24[i] if vol_ma_24[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold) with volume surge and bullish 4h trend
            if (rsi[i] < 30 and vol_surge and 
                close[i] > ema50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 (overbought) with volume surge and bearish 4h trend
            elif (rsi[i] > 70 and vol_surge and 
                  close[i] < ema50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals