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
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, Donchian, ATR
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr14_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend_1w = ema34_1w_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        atr_val = atr14_aligned[i]
        
        if position == 0:
            # Trend filter: only long when price above weekly EMA34, short when below
            price_above_ema = close[i] > ema_trend_1w
            price_below_ema = close[i] < ema_trend_1w
            
            # Volatility filter: only trade when ATR is above its 50-period median (avoid chop)
            if i >= 50:
                atr_ma = np.nanmedian(atr14_aligned[max(start_idx, i-50):i+1])
                vol_filter = atr_val > atr_ma * 0.8  # Allow trading when volatility is reasonable
            else:
                vol_filter = True
            
            # Long: break above weekly Donchian high with trend alignment and volume
            if (price_above_ema and 
                high[i] > donch_high and 
                close[i] > donch_high and
                vol_filter):
                signals[i] = size
                position = 1
            # Short: break below weekly Donchian low with trend alignment and volume
            elif (price_below_ema and 
                  low[i] < donch_low and 
                  close[i] < donch_low and
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low or trend reverses
            if low[i] < donch_low or close[i] < ema_trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high or trend reverses
            if high[i] > donch_high or close[i] > ema_trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0