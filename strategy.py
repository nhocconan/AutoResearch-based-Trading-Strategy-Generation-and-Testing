#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 1d HTF - 1d Camarilla pivot levels with 4h volume confirmation and volatility filter
    # Uses 1d Camarilla H3/L3 levels for mean reversion in ranging markets and H4/L4 for breakouts in trending markets
    # Volume confirmation ensures institutional participation, volatility filter avoids chop
    # Target: 75-200 trades over 4 years (19-50/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for volume confirmation and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h2 = np.zeros_like(close_1d)
    camarilla_l2 = np.zeros_like(close_1d)
    camarilla_h1 = np.zeros_like(close_1d)
    camarilla_l1 = np.zeros_like(close_1d)
    camarilla_pivot = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        high_y = high_1d[i-1]
        low_y = low_1d[i-1]
        close_y = close_1d[i-1]
        range_y = high_y - low_y
        
        camarilla_pivot[i] = (high_y + low_y + close_y) / 3
        camarilla_h1[i] = camarilla_pivot[i] + range_y * 1.0/12
        camarilla_l1[i] = camarilla_pivot[i] - range_y * 1.0/12
        camarilla_h2[i] = camarilla_pivot[i] + range_y * 2.0/12
        camarilla_l2[i] = camarilla_pivot[i] - range_y * 2.0/12
        camarilla_h3[i] = camarilla_pivot[i] + range_y * 3.0/12
        camarilla_l3[i] = camarilla_pivot[i] - range_y * 3.0/12
        camarilla_h4[i] = camarilla_pivot[i] + range_y * 4.0/12
        camarilla_l4[i] = camarilla_pivot[i] - range_y * 4.0/12
    
    # Calculate 4h ATR (14-period) for volatility filter
    def calculate_atr(high, low, close, window=14):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        return pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, window=14)
    atr_ma_10 = pd.Series(atr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF/LTF indicators to 4h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(atr_ma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume_4h[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_4h[i] > 0.3 * atr_ma_10_aligned[i]
        
        # Mean reversion conditions (range-bound market)
        mean_revert_long = (close_4h[i] <= camarilla_l3_aligned[i]) and volume_confirmed and vol_filter
        mean_revert_short = (close_4h[i] >= camarilla_h3_aligned[i]) and volume_confirmed and vol_filter
        
        # Breakout conditions (trending market)
        breakout_long = (close_4h[i] >= camarilla_h4_aligned[i]) and volume_confirmed and vol_filter
        breakout_short = (close_4h[i] <= camarilla_l4_aligned[i]) and volume_confirmed and vol_filter
        
        # Exit conditions: price returns to Camarilla pivot
        exit_long = position == 1 and close_4h[i] >= camarilla_pivot_aligned[i]
        exit_short = position == -1 and close_4h[i] <= camarilla_pivot_aligned[i]
        
        # Execute signals
        if mean_revert_long or breakout_long:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif mean_revert_short or breakout_short:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_breakout_meanrev_v1"
timeframe = "4h"
leverage = 1.0