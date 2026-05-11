#!/usr/bin/env python3
name = "1h_Aggressive_Volume_Momentum"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 1d data for trend and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d RSI for momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # 1d volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 1h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h volume spike
    vol_ma_1h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_1h
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume surge and momentum conditions
        volume_surge = vol_ratio[i] > 2.0
        rsi_high = rsi_1d_aligned[i] > 60
        rsi_low = rsi_1d_aligned[i] < 40
        vol_spike_1d = vol_1d[i] > vol_ma_1d_aligned[i] * 1.5
        
        if position == 0 and in_session:
            # Long: Price close above open + volume surge + RSI bullish
            if (close[i] > prices['open'].iloc[i] and 
                volume_surge and 
                rsi_high and 
                vol_spike_1d):
                signals[i] = 0.20
                position = 1
            # Short: Price close below open + volume surge + RSI bearish
            elif (close[i] < prices['open'].iloc[i] and 
                  volume_surge and 
                  rsi_low and 
                  vol_spike_1d):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: momentum reversal or session end
            if position == 1:
                # Exit long: RSI turns bearish or session ends
                if (rsi_1d_aligned[i] < 50 or not in_session):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: RSI turns bullish or session ends
                if (rsi_1d_aligned[i] > 50 or not in_session):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals