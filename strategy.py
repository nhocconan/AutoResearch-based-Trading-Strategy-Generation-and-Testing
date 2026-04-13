#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1d HTF - Camarilla pivot levels from 1d + volume spike + ATR filter
    # Designed to capture mean reversion at key intraday levels with institutional confirmation
    # Target: 12-37 trades/year (50-150 total) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # We'll use H3/L3 for entries: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # Actually standard Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first bar
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
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
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10)
    # Also need current 12h close aligned to 1d for comparison
    # We'll use the 12h close directly in loop, but need to compare to 1d levels
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(atr_ma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h bar data
        current_close = close[i]
        
        # Volume confirmation: current volume > 1.8x 20-day average
        volume_confirmed = volume[i] > 1.8 * vol_avg_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d[i] > 0.4 * atr_ma_10_aligned[i]
        
        # Mean reversion at Camarilla H3/L3 levels
        # Long when price touches or goes below L3 with volume confirmation
        # Short when price touches or goes above H3 with volume confirmation
        touch_l3 = current_close <= camarilla_l3_aligned[i]
        touch_h3 = current_close >= camarilla_h3_aligned[i]
        
        # Entry conditions
        enter_long = touch_l3 and volume_confirmed and vol_filter
        enter_short = touch_h3 and volume_confirmed and vol_filter
        
        # Exit conditions: price returns to previous day's close (mean reversion target)
        exit_long = position == 1 and current_close >= prev_close[i]
        exit_short = position == -1 and current_close <= prev_close[i]
        
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

name = "12h_1d_camarilla_pivot_volume_atr_v1"
timeframe = "12h"
leverage = 1.0