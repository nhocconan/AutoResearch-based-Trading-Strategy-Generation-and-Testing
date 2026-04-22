#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 14-period lookback + 1d EMA50 trend filter + volume spike.
# Long when %R crosses above -20 (oversold reversal) + volume spike + price > 1d EMA50
# Short when %R crosses below -80 (overbought reversal) + volume spike + price < 1d EMA50
# Exit when %R crosses back through -50 (middle) or volume drops below 70% of average.
# Williams %R captures reversals in ranging markets and pullbacks in trends.
# Target: 15-25 trades/year to minimize fee drag while capturing mean reversion and trend continuation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R on 4h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r[i]
        ema50 = ema50_aligned[i]
        
        # Williams %R crossovers
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        wr_cross_above_20 = wr > -20 and wr_prev <= -20
        wr_cross_below_80 = wr < -80 and wr_prev >= -80
        wr_cross_above_50 = wr > -50 and wr_prev <= -50
        wr_cross_below_50 = wr < -50 and wr_prev >= -50
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: %R crosses above -20 (from oversold) + volume spike + price > EMA50
            if wr_cross_above_20 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: %R crosses below -80 (from overbought) + volume spike + price < EMA50
            elif wr_cross_below_80 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: %R crosses back through -50 or volume drops significantly
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when %R crosses below -50 (losing momentum) or volume dries up
                if wr_cross_below_50 or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when %R crosses above -50 (gaining momentum) or volume dries up
                if wr_cross_above_50 or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_EMA50_Volume"
timeframe = "4h"
leverage = 1.0