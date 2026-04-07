#!/usr/bin/env python3
"""
1h_rsi_momentum_4h1d_trend_volume_v1
Hypothesis: On 1-hour timeframe, use RSI(14) momentum signals filtered by 4h EMA(50) trend and 1d volume confirmation. RSI > 60 with bullish 4h trend = long, RSI < 40 with bearish 4h trend = short. Volume must be above average to confirm institutional participation. This captures momentum in trending markets while avoiding chop. Designed for 60-150 total trades over 4 years (~15-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_momentum_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(50, 20), n):
        # Skip if data not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 1h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to hourly: divide by 24
        vol_threshold = vol_ma_1d_aligned[i] / 24.0 * 1.5
        vol_ok = volume[i] > vol_threshold
        
        if position == 1:  # Long position
            # Exit: RSI < 50 (momentum fading) or trend change
            if rsi_values[i] < 50 or ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI > 50 (momentum fading) or trend change
            if rsi_values[i] > 50 or ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Long: RSI > 60 with bullish 4h trend
                if rsi_values[i] > 60 and ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                    position = 1
                    signals[i] = 0.20
                # Short: RSI < 40 with bearish 4h trend
                elif rsi_values[i] < 40 and ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                    position = -1
                    signals[i] = -0.20
    
    return signals