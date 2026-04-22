#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) overbought/oversold with volume spike and 1d trend filter.
# Long when %R < -80 (oversold) + volume spike + price > 1d EMA50
# Short when %R > -20 (overbought) + volume spike + price < 1d EMA50
# Exit when %R crosses back above -50 (for longs) or below -50 (for shorts)
# Williams %R identifies reversals in ranging markets; works in both bull (buy dips) and bear (sell rallies).
# Target: 20-30 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14) calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values range from -100 to 0
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) + volume spike + price > EMA50
            if wr < -80 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) + volume spike + price < EMA50
            elif wr > -20 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses above -50 (momentum fading)
                if wr > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses below -50 (momentum fading)
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_OversoldOverbought_Volume_EMA50"
timeframe = "12h"
leverage = 1.0