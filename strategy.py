#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with weekly MA50 trend filter and volume confirmation
# Long when price breaks above upper BB + price > weekly MA50 + volume spike
# Short when price breaks below lower BB + price < weekly MA50 + volume spike
# Exit when price returns to opposite BB or trend reverses
# Designed for low trade frequency (~10-25/year) with strong edge in both bull and bear markets
# Uses Bollinger Bands for volatility-based support/resistance and weekly MA50 for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Bollinger Bands and weekly data for MA50
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume = df_1d['volume'].values
    
    # Calculate Bollinger Bands (20-period, 2 std dev)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate weekly MA50 for trend filter
    ma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    
    # Align indicators to daily timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ma_50_1w_aligned = align_htf_to_ltf(prices, df_1d, ma_50_1w)
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ma_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        ma_50 = ma_50_1w_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper BB + uptrend + volume spike
            if price > upper and price > ma_50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB + downtrend + volume spike
            elif price < lower and price < ma_50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite BB or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to lower BB or trend turns down
                if price < lower or price < ma_50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to upper BB or trend turns up
                if price > upper or price > ma_50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_BollingerBand_20_2_WeeklyMA50_Volume"
timeframe = "1d"
leverage = 1.0