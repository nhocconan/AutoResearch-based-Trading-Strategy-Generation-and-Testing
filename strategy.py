#!/usr/bin/env python3
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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w Donchian channel (20) - major trend structure
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA20 - trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1w ATR14 - volatility filter
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Trend filter: price above/below 1w EMA20
        trend_up = close[i] > ema_20_1w_aligned[i]
        trend_down = close[i] < ema_20_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_1w_aligned[i] > 0.008 * close[i]  # ATR > 0.8% of price
        
        # Entry conditions - balanced for daily timeframe
        # Long: upward breakout + uptrend + vol filter
        long_entry = breakout_up and trend_up and vol_filter
        # Short: downward breakout + downtrend + vol filter
        short_entry = breakout_down and trend_down and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA20_Volume_Filter"
timeframe = "1d"
leverage = 1.0