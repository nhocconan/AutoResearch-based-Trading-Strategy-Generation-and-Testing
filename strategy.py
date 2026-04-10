#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Long: 6h price > Ichimoku Cloud (Senkou Span A & B) + 1d EMA(50) > EMA(200) + 6h volume > 1.5x 20-period MA
# - Short: 6h price < Ichimoku Cloud + 1d EMA(50) < EMA(200) + 6h volume > 1.5x 20-period MA
# - Exit: Price crosses Tenkan-Kijun (TK) cross in opposite direction
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# - Ichimoku Cloud provides dynamic support/resistance; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation filters low-participation breakouts, reducing false signals

name = "6h_1d_ichimoku_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 250:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components for 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # Calculate 1d EMA(50) and EMA(200) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 6h volume moving average (20-period)
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup period (need at least 100 for Ichimoku and EMA200)
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get aligned 1d data for current 6h bar (completed 1d bar)
        ema_50_current = ema_50_aligned[i]
        ema_200_current = ema_200_aligned[i]
        
        # Get Ichimoku values for current bar
        tenkan_current = tenkan_sen[i]
        kijun_current = kijun_sen[i]
        senkou_a_current = senkou_a[i]
        senkou_b_current = senkou_b[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_current, senkou_b_current)
        lower_cloud = min(senkou_a_current, senkou_b_current)
        
        # Trend condition: EMA(50) > EMA(200) for uptrend, EMA(50) < EMA(200) for downtrend
        uptrend = ema_50_current > ema_200_current
        downtrend = ema_50_current < ema_200_current
        
        # Volume spike condition: current 6h volume > 1.5x 20-period MA
        volume_spike = volume_6h[i] > 1.5 * volume_ma_20_6h[i]
        
        # TK Cross conditions
        tk_bullish = tenkan_current > kijun_current  # Tenkan crosses above Kijun
        tk_bearish = tenkan_current < kijun_current  # Tenkan crosses below Kijun
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Cloud + uptrend + volume spike + TK bullish
            if (close_price > upper_cloud and uptrend and volume_spike and tk_bullish):
                position = 1
                signals[i] = 0.25
            # Short entry: Price < Cloud + downtrend + volume spike + TK bearish
            elif (close_price < lower_cloud and downtrend and volume_spike and tk_bearish):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when TK cross reverses (opposite direction)
            if position == 1:  # Long position
                if tk_bearish:  # Exit long when Tenkan crosses below Kijun
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if tk_bullish:  # Exit short when Tenkan crosses above Kijun
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals