#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA for trend (no look-ahead)
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ATR for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_1w = ema_1w_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price above weekly EMA + volume spike
            if price > ema_1w and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below weekly EMA + volume spike
            elif price < ema_1w and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: price crosses back below/above weekly EMA or ATR stop
            ema_cross = (position == 1 and price < ema_1w) or (position == -1 and price > ema_1w)
            stop_loss = (position == 1 and price < entry_price - 2.5 * atr_val) or \
                        (position == -1 and price > entry_price + 2.5 * atr_val)
            
            if ema_cross or stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA21_VolumeBreakout_ATRStop"
timeframe = "1d"
leverage = 1.0