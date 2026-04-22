#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with weekly trend filter and volume confirmation
# Long when price breaks above upper BB + weekly EMA50 trend up + volume spike
# Short when price breaks below lower BB + weekly EMA50 trend down + volume spike
# Exit when price returns to middle BB or volatility expands
# Designed for low trade frequency (~15-35/year) by requiring Bollinger squeeze (low volatility breakout)
# Works in bull markets via breakouts and bear markets via mean reversion in squeeze conditions

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Bollinger Bands (20, 2.0) on 6h data
    close = prices['close'].values
    bb_period = 20
    bb_std = 2.0
    
    # Middle band (SMA)
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    # Standard deviation
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (std_dev * bb_std)
    lower_bb = sma_20 - (std_dev * bb_std)
    
    # Bollinger Band Width for squeeze detection (normalized by middle band)
    bb_width = (upper_bb - lower_bb) / sma_20
    # Squeeze condition: BB width below 20-period average of BB width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_dev[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        middle = sma_20[i]
        ema_weekly = ema_50_1w_aligned[i]
        squeeze = squeeze_condition[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper BB + weekly uptrend + volume spike + squeeze
            if price > upper and price > ema_weekly and vol_spike and squeeze:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB + weekly downtrend + volume spike + squeeze
            elif price < lower and price < ema_weekly and vol_spike and squeeze:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to middle BB or volatility expands (squeeze ends)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle BB or squeeze ends
                if price <= middle or not squeeze:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle BB or squeeze ends
                if price >= middle or not squeeze:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_BollingerSqueeze_WeeklyEMA50_Volume"
timeframe = "6h"
leverage = 1.0