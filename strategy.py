#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h ATR(14) for volatility filter
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA(50) for trend direction
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    high_max_20_aligned = align_htf_to_ltf(prices, df_12h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    # 6h price data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in 12h indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or
            np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_12h_aligned[i]
        atr_val = atr_14_12h_aligned[i]
        upper_donchian = high_max_20_aligned[i]
        lower_donchian = low_min_20_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: trade only when volatility is elevated (above 30th percentile)
        vol_filter = atr_val > np.nanpercentile(atr_14_12h_aligned[:i+1], 30)
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper + uptrend + elevated volatility
            if price > upper_donchian and price > ema_50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower + downtrend + elevated volatility
            elif price < lower_donchian and price < ema_50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 12h Donchian lower or trend reverses
            if price < lower_donchian or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 12h Donchian upper or trend reverses
            if price > upper_donchian or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_DonchianBreakout_EMA50_Trend_VolFilter"
timeframe = "6h"
leverage = 1.0