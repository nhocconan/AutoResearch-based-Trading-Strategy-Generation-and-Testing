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
    
    # Calculate 1d ATR(14) for volatility
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d True Range for volatility filter
    tr_series = pd.Series(tr)
    tr_avg = tr_series.rolling(window=14, min_periods=14).mean().values
    tr_avg_aligned = align_htf_to_ltf(prices, df_1d, tr_avg)
    
    # Calculate 24-period average volume for confirmation (1d * 24 = 24h = 1 day)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(tr_avg_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: current TR > 1.5x average TR (high volatility)
        vol_filter = tr_avg_aligned[i] > 0 and tr_series.iloc[i] > (tr_avg_aligned[i] * 1.5)
        
        # Volume confirmation: current volume > 2x average volume
        vol_confirm = vol > (vol_avg[i] * 2.0) if not np.isnan(vol_avg[i]) else False
        
        # Momentum: price change over 6 periods
        if i >= 6:
            price_change = (close[i] - close[i-6]) / close[i-6]
        else:
            price_change = 0
        
        if position == 0:
            # Long: high volatility + high volume + positive momentum
            if vol_filter and vol_confirm and price_change > 0.005:
                position = 1
                signals[i] = position_size
            # Short: high volatility + high volume + negative momentum
            elif vol_filter and vol_confirm and price_change < -0.005:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: volatility drops or momentum reverses
            if not vol_filter or price_change < -0.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: volatility drops or momentum reverses
            if not vol_filter or price_change > 0.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_HighVol_Momentum_Breakout"
timeframe = "6h"
leverage = 1.0