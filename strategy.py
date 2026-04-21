#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_Trend_V1
Hypothesis: 6h Ichimoku system with TK cross + cloud filter from 1d HTF. Long when Tenkan > Kijun AND price > 1d cloud top AND price > 1d Kumo twist bullish. Short when Tenkan < Kijun AND price < 1d cloud bottom AND price < 1d Kumo twist bearish. Uses volume confirmation (>1.3x 20-period volume MA) to reduce whipsaws. ATR-based stop via signal=0 when price moves 2.5*ATR against position. Designed for medium frequency (target: 12-37 trades/year) to work in both bull/bear via cloud trend filter. Ichimoku's multi-line structure provides inherent trend/momentum/strength confirmation, reducing false breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku components)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === 1d Ichimoku components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # Cloud top/bottom and Kumo twist (bullish when Senkou A > Senkou B)
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    kumo_twist_bullish_1d = senkou_a_1d > senkou_b_1d
    kumo_twist_bearish_1d = senkou_a_1d < senkou_b_1d
    
    # Align 1d Ichimoku to 6h timeframe (completed 1d candles only)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish_1d.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish_1d.astype(float))
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # 6h Tenkan and Kijun for TK cross
    period9_high_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals (bullish when Tenkan > Kijun)
    tk_bullish_6h = tenkan_6h > kijun_6h
    tk_bearish_6h = tenkan_6h < kijun_6h
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) 
            or np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i])
            or np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i])
            or np.isnan(tk_bullish_6h[i]) or np.isnan(tk_bearish_6h[i])
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: TK bullish + price above cloud + bullish Kumo twist + volume
            if (tk_bullish_6h[i] and 
                price > cloud_top_1d_aligned[i] and 
                kumo_twist_bullish_aligned[i] > 0.5 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TK bearish + price below cloud + bearish Kumo twist + volume
            elif (tk_bearish_6h[i] and 
                  price < cloud_bottom_1d_aligned[i] and 
                  kumo_twist_bearish_aligned[i] > 0.5 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: TK bearish OR price below cloud OR Kumo twist bearish OR loss of volume
            elif (tk_bearish_6h[i] or 
                  price < cloud_top_1d_aligned[i] or 
                  kumo_twist_bullish_aligned[i] < 0.5 or 
                  not vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: TK bullish OR price above cloud OR Kumo twist bullish OR loss of volume
            elif (tk_bullish_6h[i] or 
                  price > cloud_bottom_1d_aligned[i] or 
                  kumo_twist_bearish_aligned[i] < 0.5 or 
                  not vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_Trend_V1"
timeframe = "6h"
leverage = 1.0