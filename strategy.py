#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R (14) + 1w EMA50 Trend + Volume Confirmation
# Long when Williams %R crosses above -50, price > 1w EMA50, and 1d volume > 1.5x 20-day average
# Short when Williams %R crosses below -50, price < 1w EMA50, and 1d volume > 1.5x 20-day average
# Exit when Williams %R crosses back below -80 (for longs) or above -20 (for shorts)
# Williams %R identifies overbought/oversold conditions; EMA50 filters trend direction
# Volume confirms conviction, reducing false signals
# Target: 10-25 trades/year by requiring Williams %R cross + trend alignment + volume spike

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high - close_1d) / (highest_high - lowest_low))
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Price and volume arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        price = close[i]
        ema50_val = ema50_1w_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Williams %R crosses above -50, price > EMA50, volume confirmation
            if i > 14 and williams_r_aligned[i-1] <= -50 and wr > -50 and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50, price < EMA50, volume confirmation
            elif i > 14 and williams_r_aligned[i-1] >= -50 and wr < -50 and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R crosses below -80 (overbought)
                if wr < -80:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R crosses above -20 (oversold)
                if wr > -20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_14_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0