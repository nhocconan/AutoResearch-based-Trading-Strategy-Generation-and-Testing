#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Reversal + 1w EMA Trend Filter + Volume Confirmation
# Long when Williams %R crosses above -80 from below and price > 1w EMA34 and volume > 1.5x 24-period average
# Short when Williams %R crosses below -20 from above and price < 1w EMA34 and volume > 1.5x 24-period average
# Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short)
# Williams %R identifies overbought/oversold conditions for mean reversion
# 1w EMA filter ensures we trade with the higher-timeframe trend
# Volume confirmation avoids false reversals in low-volume conditions
# Target: 15-30 trades/year by requiring all three conditions to align

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume moving average (24-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=24, min_periods=24).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 1.5x 24-period average
        vol_ma = vol_ma_1d_aligned[i]
        # For 12h timeframe, 1 bar = 12 hours = 0.5 days
        # So we need to map 12h bar to corresponding 1d volume
        # Using integer division: each 1d contains 2 12h bars
        d_idx = i // 2
        if d_idx >= len(df_1d):
            d_idx = len(df_1d) - 1
        volume = df_1d['volume'].iloc[d_idx]
        volume_confirm = volume > 1.5 * vol_ma
        
        # Trend filter: price relative to 1w EMA34
        price_above_ema = price > ema_34_1w_aligned[i]
        price_below_ema = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND price > EMA AND volume confirmation
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and
                price_above_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND price < EMA AND volume confirmation
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and
                  price_below_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses below -20 (overbought)
                if williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses above -80 (oversold)
                if williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1wEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0