#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(20) pullback to 4h EMA(50) with 1d trend filter and volume confirmation
# Uses discrete sizing 0.20 to limit fee drag. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
# 4h EMA50 defines intermediate trend, 1d EMA50 defines major trend, 1h EMA20 provides entry timing.

name = "1h_EMA20_Pullback_4hEMA50_1dEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA(20) for entry timing
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h EMA(50) for intermediate trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(50) for major trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # DatetimeIndex already, no conversion needed
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50, 50, 20)  # warmup for EMA20, EMA50_4h, EMA50_1d, vol_ma_20
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_20_1h[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_20 = ema_20_1h[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish: Price above both EMAs, pullback to 1h EMA20
                if curr_close > curr_ema_50_4h and curr_close > curr_ema_50_1d:
                    if curr_low <= curr_ema_20 and curr_close > curr_ema_20:
                        signals[i] = 0.20
                        position = 1
                        entry_price = curr_close
                # Bearish: Price below both EMAs, rally to 1h EMA20
                elif curr_close < curr_ema_50_4h and curr_close < curr_ema_50_1d:
                    if curr_high >= curr_ema_20 and curr_close < curr_ema_20:
                        signals[i] = -0.20
                        position = -1
                        entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: close below 1h EMA20 OR loses 4h trend
            if curr_close < curr_ema_20 or curr_close < curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: close above 1h EMA20 OR loses 4h trend
            if curr_close > curr_ema_20 or curr_close > curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals