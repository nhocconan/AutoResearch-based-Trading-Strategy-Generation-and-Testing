#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R overbought/oversold with volume confirmation and 12h EMA50 trend filter.
# Long when Williams %R < -80 (oversold) + volume spike + price > 12h EMA50
# Short when Williams %R > -20 (overbought) + volume spike + price < 12h EMA50
# Exit when Williams %R returns to -50 level or volume drops below 70% of average.
# Williams %R captures mean reversion in both bull and bear markets; volume confirms strength; 12h EMA50 filters trend.
# Target: 15-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Williams %R and EMA50
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) + volume spike + price > EMA50
            if wr < -80 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) + volume spike + price < EMA50
            elif wr > -20 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to -50 or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R rises above -50 or volume drops
                if wr > -50 or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R falls below -50 or volume drops
                if wr < -50 or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_OversoldOverbought_Volume_12hEMA50"
timeframe = "4h"
leverage = 1.0