#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Pullback_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h: Trend filter (EMA50) ===
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d: Trend filter (EMA50) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_4h_val = ema50_4h_aligned[i]
        ema_1d_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_4h_val) or np.isnan(ema_1d_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend on both timeframes + volume confirmation + pullback entry
            if (close_val > ema_4h_val and           # Price above 4h EMA50 (uptrend)
                close_val > ema_1d_val and           # Price above 1d EMA50 (uptrend)
                low_val < ema_4h_val and             # Pullback to 4h EMA (entry opportunity)
                vol_ratio_val > 1.5):                # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Short: Downtrend on both timeframes + volume confirmation + pullback entry
            elif (close_val < ema_4h_val and         # Price below 4h EMA50 (downtrend)
                  close_val < ema_1d_val and         # Price below 1d EMA50 (downtrend)
                  high_val > ema_4h_val and          # Pullback to 4h EMA (entry opportunity)
                  vol_ratio_val > 1.5):              # Volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or loss of momentum
            if (close_val < ema_4h_val or            # Price below 4h EMA50
                close_val < ema_1d_val or            # Price below 1d EMA50
                vol_ratio_val < 0.8):                # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal or loss of momentum
            if (close_val > ema_4h_val or            # Price above 4h EMA50
                close_val > ema_1d_val or            # Price above 1d EMA50
                vol_ratio_val < 0.8):                # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals