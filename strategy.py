#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily volume confirmation and ATR-based trend filter.
# Long when price breaks above 20-period high AND volume > 1.5x daily average volume AND ATR(14) > EMA(ATR, 20) (trending market)
# Short when price breaks below 20-period low AND volume > 1.5x daily average volume AND ATR(14) > EMA(ATR, 20)
# Exit when price crosses the opposite Donchian band or ATR drops below EMA(ATR, 20) (trend weakening)
# Uses Donchian for trend structure, volume for confirmation, ATR/EMA for trend strength filter.
# Target: 15-35 trades/year per symbol.

name = "12h_Donchian_Volume_ATRTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume and ATR calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily average volume (20-period SMA)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # EMA of ATR(20) for trend strength filter
    atr_ema = pd.Series(atr_14).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align ATR and its EMA to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ema_aligned = align_htf_to_ltf(prices, df_1d, atr_ema)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and ATR EMA need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = high_max[i]
        lower_band = low_min[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        atr = atr_14_aligned[i]
        atr_ema_val = atr_ema_aligned[i]
        
        # Trend strength filter: ATR > EMA(ATR) indicates trending market
        trending_market = atr > atr_ema_val
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above upper band + volume + trend
            if price > upper_band and volume_confirmed and trending_market:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + volume + trend
            elif price < lower_band and volume_confirmed and trending_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower band OR trend weakens
            if price < lower_band or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band OR trend weakens
            if price > upper_band or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals