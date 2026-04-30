#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud with TK cross and volume confirmation
# Ichimoku cloud provides dynamic support/resistance and trend direction.
# Tenkan-Kijun cross signals momentum shifts, with price above/below cloud filtering trend.
# Volume spike confirms institutional participation in the breakout.
# Designed for low trade frequency (12-37/year) to minimize fee drag in both bull and bear markets.
# Uses 6h timeframe with 1d HTF for Ichimoku calculation and trend filter.

name = "6h_Ichimoku_TK_Cross_CloudFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 52  # warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: TK cross bullish + price above cloud
                if curr_tenkan > curr_kijun and curr_close > cloud_top:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: TK cross bearish + price below cloud
                elif curr_tenkan < curr_kijun and curr_close < cloud_bottom:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: price breaks below cloud bottom
            if curr_close < cloud_bottom:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x cloud width above cloud top
            elif curr_close >= cloud_top + 1.5 * (cloud_top - cloud_bottom):
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price breaks above cloud top
            if curr_close > cloud_top:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x cloud width below cloud bottom
            elif curr_close <= cloud_bottom - 1.5 * (cloud_top - cloud_bottom):
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals