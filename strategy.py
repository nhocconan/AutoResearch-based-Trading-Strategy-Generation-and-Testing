#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions - extremes often precede reversals
# 1d EMA34 provides medium-term trend filter to trade with higher timeframe momentum
# Volume spike (>2.0x average) confirms institutional participation at turning points
# Works in bull/bear: extremes occur in all markets, volume confirms legitimacy, trend filter reduces counter-trend trades
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag
# Discrete position sizing: 0.25 for entries, 0.0 for flat

name = "4h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r[highest_high == lowest_low] = -50
    
    # Williams %R extremes: >80 = overbought, <20 = oversold
    williams_overbought = williams_r > 80
    williams_oversold = williams_r < 20
    
    # Volume confirmation: volume > 2.0x 20-period average (tight to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 14, 20, 34)  # warmup for Williams %R (14), volume MA (20), EMA (34)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_williams_ob = williams_overbought[i]
        curr_williams_os = williams_oversold[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on Williams extreme with volume spike and trend filter alignment
            if curr_volume_spike:
                # Bullish reversal: Williams %R oversold (<20) + price above 1d EMA34
                if curr_williams_os and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R overbought (>80) + price below 1d EMA34
                elif curr_williams_ob and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral zone (>50) or opposite extreme (>80)
            if curr_williams_r > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral zone (<50) or opposite extreme (<20)
            if curr_williams_r < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals