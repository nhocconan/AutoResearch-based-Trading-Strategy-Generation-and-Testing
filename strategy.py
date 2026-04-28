#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for long-term trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 - long-term trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily ATR14 for volatility normalization
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(atr_14[i]):
            signals[i] = 0.0
            continue
        
        # Normalized price position relative to weekly EMA200
        price_to_ema = (close[i] - ema_200_1w_aligned[i]) / atr_14[i]
        
        # Volume surge detection (2x average volume)
        vol_ma = np.mean(volume[max(0, i-20):i+1]) if i >= 20 else volume[i]
        volume_surge = volume[i] > 2.0 * vol_ma
        
        # Entry conditions
        # Long: price significantly above weekly EMA200 + volume surge
        long_entry = price_to_ema > 1.5 and volume_surge
        # Short: price significantly below weekly EMA200 + volume surge
        short_entry = price_to_ema < -1.5 and volume_surge
        
        # Exit conditions: return to mean or opposite extreme
        long_exit = price_to_ema < 0.5
        short_exit = price_to_ema > -0.5
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Close long
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.25   # Close short
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA200_VolumeSurge_MeanReversion"
timeframe = "1d"
leverage = 1.0