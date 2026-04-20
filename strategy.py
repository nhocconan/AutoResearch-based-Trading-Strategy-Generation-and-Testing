#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Keltner_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate EMA20 and ATR(10) for Keltner Channels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA20 of close
    close_series = pd.Series(close_1d)
    ema20_1d = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR(10) for channel width
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channels: Upper = EMA20 + 2*ATR, Lower = EMA20 - 2*ATR
    keltner_upper_1d = ema20_1d + 2 * atr_1d
    keltner_lower_1d = ema20_1d - 2 * atr_1d
    
    # Align Keltner channels to 4h
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    
    # === 4h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # ATR(14) for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        upper = keltner_upper_aligned[i]
        lower = keltner_lower_aligned[i]
        current_ema50 = ema50[i]
        current_atr = atr[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(current_ema50) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8x 20-period 4h average volume
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_condition = current_volume > 1.8 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long: break above upper Keltner band with volume AND above EMA50 (uptrend)
            if current_close > upper and vol_condition and current_close > current_ema50:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: break below lower Keltner band with volume AND below EMA50 (downtrend)
            elif current_close < lower and vol_condition and current_close < current_ema50:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price falls below lower Keltner band OR stop loss
            if current_close < lower or current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper Keltner band OR stop loss
            if current_close > upper or current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals