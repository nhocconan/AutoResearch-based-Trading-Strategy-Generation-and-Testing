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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_1d = pd.Series(high_1d).ewm(span=1, adjust=False).mean().values  # just to get array
    donch_low_1d = pd.Series(low_1d).ewm(span=1, adjust=False).mean().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # ATR-based volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_14[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg[i] * 1.5) if not np.isnan(vol_avg[i]) else False
        
        # Donchian breakout conditions
        bullish_breakout = price > donch_high_aligned[i]
        bearish_breakout = price < donch_low_aligned[i]
        
        if position == 0:
            # Long entry: bullish breakout + volume + volatility
            if bullish_breakout and vol_confirm and vol_filter:
                position = 1
                signals[i] = position_size
            # Short entry: bearish breakout + volume + volatility
            elif bearish_breakout and vol_confirm and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish breakout or loss of momentum
            if bearish_breakout:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish breakout or loss of momentum
            if bullish_breakout:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dDonchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0