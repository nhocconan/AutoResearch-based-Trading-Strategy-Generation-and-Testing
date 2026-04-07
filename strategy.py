#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-week trend filter and volume confirmation
# Long when Tenkan-sen crosses above Kijun-sen, price above cloud, weekly trend bullish, and volume > 1.5x 6s average
# Short when Tenkan-sen crosses below Kijun-sen, price below cloud, weekly trend bearish, and volume > 1.5x 6s average
# Exit when Tenkan/Kijun cross reverses or price crosses opposite cloud boundary
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses weekly trend filter to avoid counter-trend trades
# Ichimoku works well in both trending and ranging markets when combined with volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ichimoku_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h data for Ichimoku
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high9 = pd.Series(high_6h).rolling(window=9, min_periods=9).max()
    low9 = pd.Series(low_6h).rolling(window=9, min_periods=9).min()
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high26 = pd.Series(high_6h).rolling(window=26, min_periods=26).max()
    low26 = pd.Series(low_6h).rolling(window=26, min_periods=26).min()
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high52 = pd.Series(high_6h).rolling(window=52, min_periods=52).max()
    low52 = pd.Series(low_6h).rolling(window=52, min_periods=52).min()
    senkou_b = ((high52 + low52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b.values)
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 6h volume average for confirmation
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=24, min_periods=24).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma_6h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Tenkan/Kijun cross reverses or price below cloud
            elif (tenkan_aligned[i] < kijun_aligned[i]) or (close[i] < lower_cloud):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Tenkan/Kijun cross reverses or price above cloud
            elif (tenkan_aligned[i] > kijun_aligned[i]) or (close[i] > upper_cloud):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Tenkan/Kijun cross, cloud filter, weekly trend, and volume
            # Bullish cross: Tenkan crosses above Kijun
            bullish_cross = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
            # Bearish cross: Tenkan crosses below Kijun
            bearish_cross = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
            
            # Weekly trend: price above/below EMA20
            weekly_uptrend = close[i] > ema_20_1w_aligned[i]
            weekly_downtrend = close[i] < ema_20_1w_aligned[i]
            
            # Volume confirmation
            volume_spike = volume[i] > 1.5 * volume_ma_6h_aligned[i]
            
            # Long: bullish cross, price above cloud, weekly uptrend, volume spike
            if (bullish_cross and 
                close[i] > upper_cloud and 
                weekly_uptrend and 
                volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish cross, price below cloud, weekly downtrend, volume spike
            elif (bearish_cross and 
                  close[i] < lower_cloud and 
                  weekly_downtrend and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals