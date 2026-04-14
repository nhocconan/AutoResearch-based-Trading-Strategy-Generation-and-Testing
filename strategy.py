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
    
    # Calculate 1d Donchian(20) channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))  # close shifted
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr2[0] = tr1[0]  # first period
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d Volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: current ATR < 1.5 * 1d ATR (avoid extreme volatility)
        # Since we don't have intraday ATR, use price range as proxy
        intraday_range = high[i] - low[i]
        vol_filter = intraday_range < (1.5 * atr_1d_aligned[i])
        
        # Volume filter: current volume > 0.8 * 1d volume average
        vol_confirm = vol > (0.8 * vol_sma_20_aligned[i])
        
        if position == 0:
            # Long breakout: price > 1d Donchian high + vol filters
            if price > donchian_high_aligned[i] and vol_filter and vol_confirm:
                position = 1
                signals[i] = position_size
            # Short breakout: price < 1d Donchian low + vol filters
            elif price < donchian_low_aligned[i] and vol_filter and vol_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < 1d Donchian low OR volatility spike
            if price < donchian_low_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > 1d Donchian high OR volatility spike
            if price > donchian_high_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dDonchian20_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0