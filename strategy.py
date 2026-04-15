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
    
    # Get weekly HTF data once before loop (1w for 1d primary)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period) for trend structure
    highest_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_upper_20 = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_lower_20 = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly EMA(50) for trend bias
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily volume ratio (current vs 20-day average) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_20[i]) or np.isnan(donchian_lower_20[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when weekly ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1w_aligned[i] > 0.003 * close[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = vol_ratio[i] > 1.5
        
        # Long conditions:
        # 1. Price breaks above weekly Donchian upper (20-period breakout)
        # 2. Price above weekly EMA50 (bullish bias)
        # 3. Volume confirmation
        # 4. Volatility filter
        if (close[i] > donchian_upper_20[i] and
            close[i] > ema_50_1w_aligned[i] and
            vol_confirm and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly Donchian lower (20-period breakdown)
        # 2. Price below weekly EMA50 (bearish bias)
        # 3. Volume confirmation
        # 4. Volatility filter
        elif (close[i] < donchian_lower_20[i] and
              close[i] < ema_50_1w_aligned[i] and
              vol_confirm and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_EMA50_VolumeVolFilter_v1"
timeframe = "1d"
leverage = 1.0