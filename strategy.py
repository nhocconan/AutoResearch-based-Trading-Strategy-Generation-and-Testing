#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 for long-term trend
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily data for price channels and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-day)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Daily ATR for volatility filter and position sizing
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Daily close for momentum
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    daily_return = (close_1d - close_1d_prev) / close_1d_prev
    
    # Daily price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_10_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_w = ema200_1w_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        atr_10_val = atr_10_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: above 1.5x 20-day average
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else volume[i]
        vol_filter = vol > 1.5 * vol_ma_20
        
        # Trend filter: price above/below weekly EMA200
        uptrend = price > ema200_w
        downtrend = price < ema200_w
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + volume
            if price > donch_high_val and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + downtrend + volume
            elif price < donch_low_val and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through Donchian middle or volatility collapse
            donch_mid = (donch_high_val + donch_low_val) / 2
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian middle
                if price < donch_mid:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit on breakout above Donchian middle
                if price > donch_mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0