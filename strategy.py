#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1w trend filter and volume confirmation.
Long when price above cloud AND Tenkan > Kijun AND volume > 1.5x 20-period average.
Short when price below cloud AND Tenkan < Kijun AND volume > 1.5x 20-period average.
Exit when price crosses Tenkan-Kijun line or ATR-based stoploss hits.
Uses Ichimoku from 6h timeframe, trend filter from 1w (aligned with extra delay for cloud confirmation).
Targets 12-37 trades/year per symbol (50-150 total over 4 years) with signal size 0.25.
Ichimoku provides dynamic support/resistance and trend strength, effective in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for Ichimoku calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Align 6h Ichimoku components to 6h timeframe (no additional delay needed as they're based on completed bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe with additional delay for trend confirmation
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 6h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: price above cloud AND Tenkan > Kijun AND volume spike AND 1w uptrend
            if (price > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val and
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below cloud AND Tenkan < Kijun AND volume spike AND 1w downtrend
            elif (price < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val and
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Tenkan-Kijun midpoint or ATR stoploss
                tk_mid = (tenkan_aligned[i] + kijun_aligned[i]) / 2.0
                if price < tk_mid:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Tenkan-Kijun midpoint or ATR stoploss
                tk_mid = (tenkan_aligned[i] + kijun_aligned[i]) / 2.0
                if price > tk_mid:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0