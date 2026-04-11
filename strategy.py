#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return signals
    
    # Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Ichimoku signals
        # Bullish: Price above cloud AND Tenkan > Kijun
        bullish = (price_close > senkou_a_aligned[i] and 
                   price_close > senkou_b_aligned[i] and 
                   tenkan_aligned[i] > kijun_aligned[i])
        
        # Bearish: Price below cloud AND Tenkan < Kijun
        bearish = (price_close < senkou_a_aligned[i] and 
                   price_close < senkou_b_aligned[i] and 
                   tenkan_aligned[i] < kijun_aligned[i])
        
        # Trading logic
        if bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and not (bullish and vol_confirm):
            position = 0
            signals[i] = 0.0
        elif position == -1 and not (bearish and vol_confirm):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Ichimoku cloud strategy with daily trend filter and volume confirmation.
# Enters long when price is above the cloud (Senkou Span A & B) AND Tenkan > Kijun with volume confirmation.
# Enters short when price is below the cloud AND Tenkan < Kijun with volume confirmation.
# Exits when the trend condition fails (price crosses cloud or Tenkan/Kijun cross reverses).
# Uses daily Ichimoku for higher timeframe trend structure, reducing false signals in sideways markets.
# Volume filter ensures participation during institutional interest periods.
# Designed to work in both bull and bear markets by capturing sustained trends in either direction.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity with fee minimization.
# Position size 0.25 balances risk exposure with return potential in volatile crypto markets.