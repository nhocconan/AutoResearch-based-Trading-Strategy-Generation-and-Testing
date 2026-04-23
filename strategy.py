#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume > 2.0x average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume > 2.0x average.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR trend reverses.
Uses 12h timeframe for lower trade frequency (target: 50-150 trades over 4 years) and Williams %R for mean reversion in ranging markets.
1d EMA34 provides smooth trend filter. Volume spike ensures high-conviction entries.
Designed to work in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R (14-period) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = williams_r_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume spike
            if (wr_val < -80.0 and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (wr_val > -20.0 and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 OR price breaks below 1d EMA34 (trend reversal)
                if wr_val > -50.0 or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 OR price breaks above 1d EMA34 (trend reversal)
                if wr_val < -50.0 or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0