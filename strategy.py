#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_volume_squeeze"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner channels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Keltner Channels (20-period, 2.0 ATR multiplier)
    atr_period = 20
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    
    # Volume filter: 20-period average on daily volume
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_strong = volume_1d > vol_ma_1d
    
    # Align to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    volume_strong_aligned = align_htf_to_ltf(prices, df_1d, volume_strong)
    
    # 4h price action for entry
    # Calculate 4h ATR for stop loss
    tr_4h1 = high[1:] - low[1:]
    tr_4h2 = np.abs(high[1:] - close[:-1])
    tr_4h3 = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))])
    atr_4h = pd.Series(tr_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_strong_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strong condition
        vol_ok = volume_strong_aligned[i]
        
        # Keltner squeeze breakout signals
        # Long: Close breaks above upper Keltner band with volume
        long_signal = close[i] > upper_keltner_aligned[i] and vol_ok
        # Short: Close breaks below lower Keltner band with volume
        short_signal = close[i] < lower_keltner_aligned[i] and vol_ok
        
        # Exit when price returns to middle (EMA20) - mean reversion tendency
        exit_long = close[i] < ema_20_aligned[i]
        exit_short = close[i] > ema_20_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals