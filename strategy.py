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
    
    # Calculate 1d Donchian channels (20-period) for breakout signals
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate 1d ATR(14) for volatility filter and position sizing
    tr1 = high_series - low_series
    tr2 = abs(high_series - close.shift(1))
    tr3 = abs(low_series - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1d average volume for confirmation
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_14_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average daily volume
        vol_confirm = vol > (vol_avg_aligned[i] * 1.5) if not np.isnan(vol_avg_aligned[i]) else False
        
        # Breakout signals: price breaks above/below 1d Donchian channels
        breakout_long = price > donch_high_aligned[i]
        breakout_short = price < donch_low_aligned[i]
        
        if position == 0:
            # Long entry: bullish breakout + volume confirmation + volatility filter
            if breakout_long and vol_confirm and vol_filter:
                position = 1
                signals[i] = position_size
            # Short entry: bearish breakout + volume confirmation + volatility filter
            elif breakout_short and vol_confirm and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian channel
            if price < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian channel
            if price > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian20_Breakout_Volume"
timeframe = "12h"
leverage = 1.0