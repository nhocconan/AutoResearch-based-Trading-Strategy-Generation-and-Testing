#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_ichimoku_trend_follow_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return signals
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Align Ichimoku components to 12h timeframe (with proper delay for completed weekly bars)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Volume confirmation: volume > 1.5x 20-period average on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Bullish trend: price above both Senkou spans AND Tenkan > Kijun
        bullish_trend = (price_high > senkou_span_a_aligned[i] and 
                        price_high > senkou_span_b_aligned[i] and
                        tenkan_sen_aligned[i] > kijun_sen_aligned[i])
        
        # Bearish trend: price below both Senkou spans AND Tenkan < Kijun
        bearish_trend = (price_low < senkou_span_a_aligned[i] and 
                        price_low < senkou_span_b_aligned[i] and
                        tenkan_sen_aligned[i] < kijun_sen_aligned[i])
        
        # Entry conditions with volume confirmation
        long_signal = bullish_trend and volume_confirmed
        short_signal = bearish_trend and volume_confirmed
        
        # Exit when trend reverses
        exit_long = position == 1 and not bullish_trend
        exit_short = position == -1 and not bearish_trend
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Ichimoku trend following on 12h timeframe with weekly trend filter.
# Uses weekly Ichimoku Cloud to determine primary trend direction (bullish/bearish).
# Enters long when price is above the cloud AND Tenkan-sen > Kijun-sen with volume confirmation (>1.5x avg volume).
# Enters short when price is below the cloud AND Tenkan-sen < Kijun-sen with volume confirmation.
# Exits when the trend condition breaks (price crosses cloud or Tenkan/Kijun crossover reverses).
# The weekly Ichimoku provides a robust multi-week trend filter that works in both bull and bear markets.
# Volume confirmation ensures trades occur with institutional participation.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift on 12h timeframe.