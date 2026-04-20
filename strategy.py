#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h trend filter
# - Long when price breaks above Donchian(20) high + volume > 1.5x average + 12h EMA34 > EMA89
# - Short when price breaks below Donchian(20) low + volume > 1.5x average + 12h EMA34 < EMA89
# - Exit when price crosses back through Donchian(20) midline or volume drops
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Donchian provides clear breakout levels, volume confirms conviction, 12h EMA filters trend
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 and EMA89 on 12h timeframe
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    # Calculate Donchian(20) channels on 4h price data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Calculate volume confirmation: current volume > 1.5x 20-period average
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_4h / avg_volume_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(89, n):  # Start after EMA89 warmup
        # Skip if NaN in indicators
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(ema89_12h_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol_ratio = volume_ratio[i]
        ema34 = ema34_12h_aligned[i]
        ema89 = ema89_12h_aligned[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        midline = donchian_mid[i]
        
        if position == 0:
            # Long entry: break above upper channel + volume confirmation + 12h uptrend
            if (price > upper_channel and vol_ratio > 1.5 and ema34 > ema89):
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower channel + volume confirmation + 12h downtrend
            elif (price < lower_channel and vol_ratio > 1.5 and ema34 < ema89):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midline or volume drops significantly
            if price < midline or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midline or volume drops significantly
            if price > midline or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_EMA34_89_TrendFilter"
timeframe = "4h"
leverage = 1.0