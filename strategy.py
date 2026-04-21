#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_12hTrendRegime
Hypothesis: Ichimoku TK cross on 6h with cloud filter from 12h and 1d EMA50 trend regime captures momentum with controlled frequency. Uses discrete sizing (0.25) and ATR-based stoploss (2.0x) to minimize fee drag. Target: 50-150 total trades over 4 years for BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for cloud filter, 1d for trend regime)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h Ichimoku cloud (Senkou Span A/B) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_span_a_aligned[i]
        senkou_b = senkou_span_b_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Trend regime
        is_bull = price > ema_50_1d_val
        is_bear = price < ema_50_1d_val
        
        # TK cross conditions
        tk_bullish_cross = (tenkan > kijun) and (np.roll(tenkan, 1)[i] <= np.roll(kijun, 1)[i])
        tk_bearish_cross = (tenkan < kijun) and (np.roll(tenkan, 1)[i] >= np.roll(kijun, 1)[i])
        
        # Handle first bar where roll gives same value
        if i == 0:
            tk_bullish_cross = False
            tk_bearish_cross = False
        
        if position == 0:
            if is_bull:
                # Bull regime: long when TK bullish cross and price above cloud
                long_condition = tk_bullish_cross and (price > cloud_top) and vol_conf
            else:  # bear regime
                # Bear regime: short when TK bearish cross and price below cloud
                short_condition = tk_bearish_cross and (price < cloud_bottom) and vol_conf
            
            if 'long_condition' in locals() and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif 'short_condition' in locals() and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price falls below cloud (cloud break)
                elif price < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price rises above cloud (cloud break)
                elif price > cloud_top:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_12hTrendRegime"
timeframe = "6h"
leverage = 1.0