#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA Trend + Volume Spike
# Long when Williams %R < -80 (oversold), price > 1d EMA50, and 1d volume > 1.8x 20-day avg
# Short when Williams %R > -20 (overbought), price < 1d EMA50, and 1d volume > 1.8x 20-day avg
# Exit when Williams %R crosses -50 (mean reversion) or price crosses 1d EMA50
# Williams %R identifies overbought/oversold conditions, effective in ranging markets
# EMA50 filter ensures alignment with intermediate trend, avoiding counter-trend trades
# Volume spike confirms conviction, reducing false signals
# Target: 15-25 trades/year by requiring strict oversold/overbought + volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-period Williams %R
    high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    close_1d = df_1d['close'].values
    williams_r = -100 * (high_14 - close_1d) / (high_14 - low_14)
    williams_r = williams_r.values  # Convert to numpy array
    
    # Calculate 1d EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Price and volume arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
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
        
        # Get corresponding 1d volume (2 periods per day for 12h timeframe)
        vol_index = i // 2
        if vol_index >= len(df_1d):
            vol_index = len(df_1d) - 1
        volume = df_1d['volume'].iloc[vol_index] if vol_index >= 0 else df_1d['volume'].iloc[0]
        
        # Volume confirmation: current 1d volume > 1.8x 20-day average
        volume_confirm = volume > 1.8 * vol_ma if vol_index >= 20 else False
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), price > EMA50, volume confirmation
            if wr < -80 and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price < EMA50, volume confirmation
            elif wr > -20 and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R crosses above -50 or price crosses below EMA50
                if wr > -50 or price < ema50_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R crosses below -50 or price crosses above EMA50
                if wr < -50 or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0