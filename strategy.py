#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Ichimoku Cloud + Volume Spike
# - Williams %R(14) on 6h: oversold < -80 (long), overbought > -20 (short)
# - 1d Ichimoku Cloud: price above cloud (bullish bias), below cloud (bearish bias)
# - Volume confirmation: current volume > 2.0x 24-period average (6h = 4*6h = 1d)
# - Long: Williams %R < -80 AND price > 1d cloud top AND volume spike
# - Short: Williams %R > -20 AND price < 1d cloud bottom AND volume spike
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R catches mean reversions in both bull and bear markets
# - Ichimoku cloud provides higher-timeframe trend filter to avoid counter-trend trades
# - Volume spike ensures participation and filters false signals

name = "6h_1d_williamsr_ichimoku_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (no extra delay needed for cloud)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute 6h volume confirmation (24-period average = 1 day)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(volume_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price = close[i]
        volume_current = volume[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Ichimoku cloud conditions
        price_above_cloud = price > cloud_top[i]
        price_below_cloud = price < cloud_bottom[i]
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume_current > 2.0 * volume_sma_24[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold + price above cloud + volume spike
        if oversold and price_above_cloud and vol_confirm:
            enter_long = True
        
        # Short: Williams %R overbought + price below cloud + volume spike
        if overbought and price_below_cloud and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse Williams %R or loss of cloud alignment
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R rises above -50 OR price falls below cloud
            exit_long = williams_r[i] > -50 or not price_above_cloud
        elif position == -1:
            # Exit short if Williams %R falls below -50 OR price rises above cloud
            exit_short = williams_r[i] < -50 or not price_below_cloud
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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