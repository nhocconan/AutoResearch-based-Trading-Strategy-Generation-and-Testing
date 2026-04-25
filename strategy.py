#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX trend filter and volume spike filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend strength and Ichimoku cloud calculation (senkou span A/B).
- Ichimoku Components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (26/52-period).
- Trend Filter: 1d ADX > 25 to ensure trending market (avoids chop/whipsaws).
- Volume Filter: Current 6h volume > 1.8 * 20-period average 6h volume to confirm momentum.
- Entry: Long when price > Senkou Span A AND Tenkan-sen > Kijun-sen AND price > cloud AND ADX > 25 AND volume spike.
         Short when price < Senkou Span B AND Tenkan-sen < Kijun-sen AND price < cloud AND ADX > 25 AND volume spike.
- Exit: Opposite Ichimoku signal (long exits when price < cloud, short exits when price > cloud).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong trending moves with cloud confirmation while filtering weak/choppy markets.
- Works in bull markets (trend continuation up) and bear markets (trend continuation down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for Ichimoku calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 1d ADX for trend filter
    # ADX calculation: +DI, -DI, DX, then ADX
    period_adx = 14
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = (pd.Series(close_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period_adx, min_periods=period_adx).mean()
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # +DI and -DI
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period_adx, min_periods=period_adx).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period_adx, min_periods=period_adx).mean() / atr)
    
    # DX and ADX
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).rolling(window=period_adx, min_periods=period_adx).mean()
    adx_values = adx.values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20, 14*2)  # Need 52 for Senkou B, 20 for volume MA, 28 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Ichimoku conditions
        price_above_cloud = curr_close > senkou_a and curr_close > senkou_b
        price_below_cloud = curr_close < senkou_a and curr_close < senkou_b
        tenkan_above_kijun = tenkan > kijun
        tenkan_below_kijun = tenkan < kijun
        strong_trend = adx_val > 25
        
        # Exit conditions: opposite Ichimoku signal
        if position != 0:
            # Exit long: price goes below cloud
            if position == 1:
                if price_below_cloud:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price goes above cloud
            elif position == -1:
                if price_above_cloud:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku signal with trend and volume filters
        if position == 0:
            # Long: price above cloud AND Tenkan > Kijun AND strong trend AND volume spike
            long_condition = price_above_cloud and tenkan_above_kijun and strong_trend and volume_spike
            
            # Short: price below cloud AND Tenkan < Kijun AND strong trend AND volume spike
            short_condition = price_below_cloud and tenkan_below_kijun and strong_trend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0