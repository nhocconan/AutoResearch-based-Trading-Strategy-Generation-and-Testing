#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (-20/-80 levels)
# ADX > 25 filters for trending markets, < 20 for ranging
# In trending markets (ADX>25): fade extreme %R readings (mean reversion)
# In ranging markets (ADX<20): breakout continuation from %R extremes
# Volume confirmation ensures participation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_WilliamsR_Reversal_1dADXTrend_Volume"
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
    
    # Williams %R (14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on 1d
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Determine market regime from 1d ADX
            is_trending = adx_aligned[i] > 25
            is_ranging = adx_aligned[i] < 20
            
            if is_trending:
                # In trending markets: fade extreme Williams %R (mean reversion)
                # Long: %R oversold (< -80) turning up with volume
                if (williams_r[i] < -80 and 
                    williams_r[i] > williams_r[i-1] and  # Turning up
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: %R overbought (> -20) turning down with volume
                elif (williams_r[i] > -20 and 
                      williams_r[i] < williams_r[i-1] and  # Turning down
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # In ranging markets: breakout continuation from %R extremes
                # Long: %R oversold (< -80) breaking higher with volume
                if (williams_r[i] < -80 and 
                    close[i] > high[i-1] and  # Breaking above prior high
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: %R overbought (> -20) breaking lower with volume
                elif (williams_r[i] > -20 and 
                      close[i] < low[i-1] and  # Breaking below prior low
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX 20-25): no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral zone (-50) or adverse move
            if williams_r[i] > -50 or williams_r[i] < williams_r[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral zone (-50) or adverse move
            if williams_r[i] < -50 or williams_r[i] > williams_r[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals