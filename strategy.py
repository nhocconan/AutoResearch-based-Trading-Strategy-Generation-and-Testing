#!/usr/bin/env python3
"""
Hypothesis: 1h RSI Mean Reversion with 4h Trend Filter and Volume Spike.
Long when 1h RSI < 30, price > 4h EMA50, and volume > 1.5x 20-bar average in 08-20 UTC session.
Short when 1h RSI > 70, price < 4h EMA50, and volume > 1.5x 20-bar average in 08-20 UTC session.
Exit when RSI reverts to 50 or volume condition fails.
Uses 4h for trend direction (reduces whipsaw), 1h for precise entry timing.
Target: 80-120 total trades over 4 years (20-30/year). Discrete sizing 0.20 to control turnover.
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5x 20-bar average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        session_ok = in_session[i]
        ema_50 = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold, above 4h EMA50, volume spike, in session
            if rsi_val < 30 and price > ema_50 and vol_spike and session_ok:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought, below 4h EMA50, volume spike, in session
            elif rsi_val > 70 and price < ema_50 and vol_spike and session_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI reverts to 50 or volume/spike/session condition fails
            if rsi_val >= 50 or not vol_spike or not session_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI reverts to 50 or volume/spike/session condition fails
            if rsi_val <= 50 or not vol_spike or not session_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0