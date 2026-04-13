#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1d HTF - 12h Camarilla pivot breakout with 1d volume confirmation and ATR volatility filter
    # Designed to capture institutional breakouts at key daily pivot levels with volume validation
    # Target: 50-150 trades over 4 years (12-37/year) for low fee drag and good generalization
    # Works in bull/bear by using volatility filter to avoid chop and volume confirmation to avoid false breakouts
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Typical Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We'll use R3/S3 as breakout levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Calculate Camarilla levels for breakout
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)  # Resistance 3
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)  # Support 3
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)  # Resistance 4 (stronger)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)  # Support 4 (stronger)
    
    # Calculate 1d ATR (14-period) for volatility filter
    def calculate_atr(high, low, close, window=14):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        return pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, window=14)
    atr_ma_10 = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(atr_ma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume_1d[i] > 1.3 * vol_avg_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d[i] > 0.2 * atr_ma_10_aligned[i]
        
        # Breakout conditions at Camarilla levels
        breakout_up = close_1d[i] > camarilla_h3_aligned[i]
        breakout_down = close_1d[i] < camarilla_l3_aligned[i]
        
        # Strong breakout conditions (using H4 levels for confirmation)
        strong_breakout_up = close_1d[i] > camarilla_h4_aligned[i]
        strong_breakout_down = close_1d[i] < camarilla_l4_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and vol_filter
        enter_short = breakout_down and volume_confirmed and vol_filter
        
        # Exit conditions: price returns to previous day's close (pivot point)
        exit_long = position == 1 and close_1d[i] <= prev_close_1d[i]
        exit_short = position == -1 and close_1d[i] >= prev_close_1d[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_volume_atr_v1"
timeframe = "12h"
leverage = 1.0