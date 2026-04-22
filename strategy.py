#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly high/low for Donchian channel (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channel (20-week period)
    highest_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    highest_20w_aligned = align_htf_to_ltf(prices, df_1w, highest_20w)
    lowest_20w_aligned = align_htf_to_ltf(prices, df_1w, lowest_20w)
    
    # Daily ATR for volatility regime and stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA for trend filter
    ema = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(highest_20w_aligned[i]) or 
            np.isnan(lowest_20w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        ema_val = ema[i]
        upper_channel = highest_20w_aligned[i]
        lower_channel = lowest_20w_aligned[i]
        volume = prices['volume'].values[i]
        vol_threshold = vol_ma[i]
        
        # Volume confirmation
        vol_ok = volume > 1.5 * vol_threshold
        
        if position == 0 and vol_ok:
            # Long: price breaks above weekly Donchian upper + trend filter
            if price > upper_channel and price > ema_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly Donchian lower + trend filter
            elif price < lower_channel and price < ema_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: price crosses EMA (trend reversal) or volatility collapse
            trend_reverse = (position == 1 and price < ema_val) or (position == -1 and price > ema_val)
            vol_collapse = atr_val < 0.5 * atr[i-1] if i > 0 else False
            
            if trend_reverse or vol_collapse:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchianBreakout_EMA50Trend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0