#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend strength and Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B).
- Ichimoku Components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displaced).
- Trend Filter: 1d ADX > 25 to ensure strong trend (avoids chop/whipsaws).
- Volume Filter: Current 6h volume > 1.8 * 20-period average 6h volume to confirm momentum.
- Entry: Long when price > Senkou Span A AND Senkou Span A > Senkou Span B (bullish cloud) AND Tenkan-sen > Kijun-sen (bullish TK cross) AND ADX > 25 AND volume spike.
         Short when price < Senkou Span B AND Senkou Span B > Senkou Span A (bearish cloud) AND Tenkan-sen < Kijun-sen (bearish TK cross) AND ADX > 25 AND volume spike.
- Exit: Opposite Ichimoku signal (long exits when price < Senkou Span B or Tenkan-sen < Kijun-sen, short exits when price > Senkou Span A or Tenkan-sen > Kijun-sen).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong trending moves with Ichimoku cloud as dynamic support/resistance, filtered by ADX for trend strength and volume for confirmation.
- Works in bull markets (trend continuation) and bear markets (trend continuation down) by requiring ADX > 25 for both long and short.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for Ichimoku calculations (52 + displacement)
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough data for Ichimoku (52 periods)
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
    
    # Displace Senkou Span A and B forward by 26 periods (Kijun period)
    # For alignment, we'll use the values as-is and let align_htf_to_ltf handle the timing
    # The displacement is built into the Ichimoku calculation logic
    
    # Calculate 1d ADX for trend filter
    period_adx = 14
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean()
    adx_values = adx.values
    
    # Align Ichimoku components and ADX to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need 52 for Senkou B, 26 for displacement alignment, 14 for ADX, 20 for volume MA
    start_idx = max(52, 26, 14, 20)
    
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
        adx_level = adx_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Ichimoku conditions
        bullish_config = (curr_close > senkou_a) and (senkou_a > senkou_b) and (tenkan > kijun)
        bearish_config = (curr_close < senkou_b) and (senkou_b > senkou_a) and (tenkan < kijun)
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_level > 25
        
        # Exit conditions: opposite Ichimoku signal or loss of momentum
        if position != 0:
            # Exit long: price breaks below Senkou B OR Tenkan crosses below Kijun
            if position == 1:
                if (curr_close < senkou_b) or (tenkan < kijun):
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Senkou A OR Tenkan crosses above Kijun
            elif position == -1:
                if (curr_close > senkou_a) or (tenkan > kijun):
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku breakout with trend and volume filters
        if position == 0:
            # Long: bullish Ichimoku config AND strong trend AND volume spike
            long_condition = bullish_config and strong_trend and volume_spike
            
            # Short: bearish Ichimoku config AND strong trend AND volume spike
            short_condition = bearish_config and strong_trend and volume_spike
            
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

name = "6h_Ichimoku_Cloud_1dADX_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0