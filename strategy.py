#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Long when: price > 1d Ichimoku cloud (Senkou Span A/B), Tenkan > Kijun (bullish TK cross), and 6h volume > 1.5x its 20-period EMA.
# Short when: price < 1d Ichimoku cloud, Tenkan < Kijun (bearish TK cross), and 6h volume > 1.5x its 20-period EMA.
# Exit when price re-enters the cloud or TK cross reverses.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Ichimoku provides dynamic support/resistance; TK cross gives momentum; volume confirms strength; 1d cloud filters for higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # === 1d Ichimoku Cloud Components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    volume_ema_20 = pd.Series(volume_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (6h)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    volume_ema_20_aligned = align_htf_to_ltf(prices, df_6h, volume_ema_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(52, 20)  # Ichimoku needs 52 periods, volume EMA needs 20
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        volume_ema_20_val = volume_ema_20_aligned[i]
        volume_val = volume[i]
        
        # Ichimoku cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross conditions
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # Price vs cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation: current volume > 1.5x its 20-period EMA
        volume_confirm = volume_val > (volume_ema_20_val * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price re-enters cloud OR TK cross turns bearish
            if not price_above_cloud or not tk_bullish:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price re-enters cloud OR TK cross turns bullish
            if not price_below_cloud or not tk_bearish:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price above cloud AND bullish TK cross AND volume confirmation
            if price_above_cloud and tk_bullish and volume_confirm:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price below cloud AND bearish TK cross AND volume confirmation
            elif price_below_cloud and tk_bearish and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_IchimokuCloud_1dTKCross_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0