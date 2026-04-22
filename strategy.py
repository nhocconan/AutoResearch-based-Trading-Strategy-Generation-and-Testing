#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Ichimoku Cloud with 1d ADX trend filter and volume confirmation.
# Ichimoku Cloud provides dynamic support/resistance and trend direction.
# In trending markets (ADX > 25), price above/below cloud with TK cross signals strong momentum.
# Volume spike (>2x 24-period average) confirms institutional participation.
# Designed for low trade frequency (~15-25/year) to minimize fee decay.
# Works in bull markets (long when price > cloud + TK cross up) and bear markets (short when price < cloud + TK cross down).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    chikou = close_1d  # Will be handled by alignment
    
    # Calculate 14-period ADX for trend strength filter
    # True Range
    tr1 = pd.Series(high_1d).rolling(window=1).max() - pd.Series(low_1d).rolling(window=1).min()
    tr2 = abs(pd.Series(high_1d).rolling(window=1).max() - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).rolling(window=1).min() - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe (waits for 1d bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 24-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 2.0 * 24-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Cloud determination: Senkou Span A and B form the cloud
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # TK Cross: Tenkan-sen crossing Kijun-sen
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        # Price relative to cloud
        price_above_cloud = price > upper_cloud
        price_below_cloud = price < lower_cloud
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long conditions: price above cloud + TK cross up + strong trend + volume spike
            if price_above_cloud and tk_cross_up and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below cloud + TK cross down + strong trend + volume spike
            elif price_below_cloud and tk_cross_down and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price falls below cloud or TK cross down
                if not price_above_cloud or tk_cross_down:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above cloud or TK cross up
                if not price_below_cloud or tk_cross_up:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_IchimokuCloud_1dADX_Volume"
timeframe = "12h"
leverage = 1.0