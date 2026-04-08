#!/usr/bin/env python3
# 12h_keltner_channel_v1
# Hypothesis: 12h Keltner Channel breakout with weekly RSI filter and volume confirmation.
# Goes long when price closes above upper Keltner band in weekly uptrend (RSI > 50) with volume surge.
# Goes short when price closes below lower Keltner band in weekly downtrend (RSI < 50) with volume surge.
# Designed for low trade frequency (12-37/year) to avoid fee drag, works in bull/bear via weekly trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_keltner_channel_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly trend filter: RSI(14)
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Keltner Channel (20-period, ATR multiplier 2.0)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean()
    atr[:13] = np.nan  # Handle first values
    ema_middle = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_middle + 2.0 * atr.values
    lower_keltner = ema_middle - 2.0 * atr.values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = rsi_1w_aligned[i] > 50
        weekly_downtrend = rsi_1w_aligned[i] < 50
        
        # Keltner breakout signals
        breakout_up = close[i] > upper_keltner[i]
        breakout_down = close[i] < lower_keltner[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Close below middle line or trend change
            if close[i] < ema_middle[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above middle line or trend change
            if close[i] > ema_middle[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: Close above upper Keltner in weekly uptrend
                if weekly_uptrend and breakout_up:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Close below lower Keltner in weekly downtrend
                elif weekly_downtrend and breakout_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals