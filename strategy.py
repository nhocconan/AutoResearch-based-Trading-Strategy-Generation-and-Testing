#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation
# Long when Tenkan-sen > Kijun-sen (TK cross), price above Kumo (cloud), 1d close > 1d EMA200 (uptrend), and volume > 1.5x 6s average volume
# Short when Tenkan-sen < Kijun-sen (TK cross), price below Kumo (cloud), 1d close < 1d EMA200 (downtrend), and volume > 1.5x 6s average volume
# Exit when TK cross reverses or price crosses Kijun-sen
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Ichimoku for momentum/trend, 1d EMA200 for higher timeframe trend filter, volume for confirmation
# Designed to work in both bull and bear markets by requiring alignment across timeframes and volume confirmation

name = "6h_ichimoku_1d_ema200_vol_v1"
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
    high_9 = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B shifted forward by 26 periods
    # For cloud calculation, we use current Senkou spans
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6s volume average for confirmation
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_6h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: TK cross reverses (Tenkan < Kijun) or price crosses Kijun-sen
            elif tenkan_aligned[i] < kijun_aligned[i] or close[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: TK cross reverses (Tenkan > Kijun) or price crosses Kijun-sen
            elif tenkan_aligned[i] > kijun_aligned[i] or close[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: TK cross bullish (Tenkan > Kijun), price above cloud, price above EMA200 (uptrend), volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and
                close[i] > kumo_top[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: TK cross bearish (Tenkan < Kijun), price below cloud, price below EMA200 (downtrend), volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and
                  close[i] < kumo_bottom[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals