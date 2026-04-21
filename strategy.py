#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA50 Trend + Volume Spike
# Long when Williams %R crosses above -80 (oversold), price > 1d EMA50, and 1d volume > 1.5x 20-day average
# Short when Williams %R crosses below -20 (overbought), price < 1d EMA50, and 1d volume > 1.5x 20-day average
# Williams %R measures momentum: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Works in both bull (buy oversold dips) and bear (sell overbought rallies)
# Volume confirms conviction, EMA50 filters trend direction
# Target: 25-40 trades/year by requiring all three conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Calculate 1d EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        price = close[i]
        ema50_val = ema50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        # Get the corresponding 1d volume for this 4h bar
        idx_1d = i // 6  # 6 four-hour bars per day
        if idx_1d < len(df_1d):
            vol_current = df_1d['volume'].iloc[idx_1d]
            volume_confirm = vol_current > 1.5 * vol_ma
        else:
            volume_confirm = False
        
        if position == 0:
            # Long: Williams %R > -80 (coming from oversold), price > EMA50, volume confirmation
            if wr > -80 and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R < -20 (coming from overbought), price < EMA50, volume confirmation
            elif wr < -20 and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R < -80 or price crosses below EMA50
                if wr < -80 or price < ema50_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R > -20 or price crosses above EMA50
                if wr > -20 or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0