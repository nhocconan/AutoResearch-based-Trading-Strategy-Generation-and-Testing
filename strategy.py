#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period).
# Long when price > cloud and TK cross bullish, short when price < cloud and TK cross bearish.
# 1d EMA50 provides higher timeframe trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50.
# Volume confirmation requires current volume > 1.8x 20-period average to filter weak breakouts.
# Designed to capture trends while avoiding whipsaws in ranging markets.
# Targets 15-30 trades/year with strict multi-condition entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d data
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Ichimoku components on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # The Ichimoku cloud is between Senkou Span A and Senkou Span B
    # For simplicity, we use the current Senkou Span values (not shifted)
    # In practice, Senkou spans are plotted 26 periods ahead, but for signal generation
    # we use current values to represent current cloud boundaries
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou Span B calculation is valid
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_1d_aligned[i]
        
        # Cloud boundaries: higher band = max(Senkou A, Senkou B), lower band = min(Senkou A, Senkou B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # TK cross: Tenkan-sen crossing Kijun-sen
        tk_cross_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price above cloud, bullish TK cross, uptrend filter, volume spike
            if price > cloud_top and tk_cross_bullish and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below cloud, bearish TK cross, downtrend filter, volume spike
            elif price < cloud_bottom and tk_cross_bearish and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: TK cross in opposite direction or price returns to cloud
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish TK cross or price drops below cloud bottom
                if tk_cross_bearish or price < cloud_bottom:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish TK cross or price rises above cloud top
                if tk_cross_bullish or price > cloud_top:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_1dEMA_Trend_Volume"
timeframe = "6h"
leverage = 1.0