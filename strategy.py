#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) mean reversion with 1d EMA89 trend filter and volume confirmation.
# Long when %R < -80 (oversold) + price > 1d EMA89 + volume spike
# Short when %R > -20 (overbought) + price < 1d EMA89 + volume spike
# Exit when %R crosses back through -50 (mean) or volume drops.
# Williams %R is effective in ranging markets (2025+) and captures reversals in trends.
# Target: 20-40 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 90:
        return np.zeros(n)
    
    # Load 1d data for EMA89 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA89 for trend filter
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Williams %R (14) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Align 1d EMA89 to 4h
    ema89_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(89, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema89_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r[i]
        ema89 = ema89_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) + price > EMA89 + volume spike
            if wr < -80 and price > ema89 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) + price < EMA89 + volume spike
            elif wr > -20 and price < ema89 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses back through -50 (mean) or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses above -50 or volume dries up
                if wr > -50 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses below -50 or volume dries up
                if wr < -50 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dEMA89_Volume"
timeframe = "4h"
leverage = 1.0