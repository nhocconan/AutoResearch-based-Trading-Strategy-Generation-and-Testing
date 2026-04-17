#1
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R(14) + 1d EMA50 trend filter + volume spike.
# Williams %R identifies overbought/oversold conditions; reversals are more reliable in trending markets.
# EMA50 on daily timeframe filters for trend direction. Volume spike confirms momentum.
# Designed for mean reversion within trend, effective in both bull and bear markets.
# Target: 15-25 trades/year to stay within optimal range for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 14-period Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high_14 - lowest_low_14
    williams_r = np.where(hh_ll != 0, ((highest_high_14 - close_1d) / hh_ll) * -100, -50.0)
    
    # Align 1d indicators to 12h
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    williams_r_12h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need EMA50 and Williams %R lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(williams_r_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.8x average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Williams %R levels: oversold < -80, overbought > -20
        williams_oversold = williams_r_12h[i] < -80
        williams_overbought = williams_r_12h[i] > -20
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        if position == 0:
            # Long: Williams %R oversold + price above EMA + volume spike
            if (williams_oversold and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought + price below EMA + volume spike
            elif (williams_overbought and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum fading) OR price below EMA
            if (williams_r_12h[i] > -50) or (close[i] < ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum fading) OR price above EMA
            if (williams_r_12h[i] < -50) or (close[i] > ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0