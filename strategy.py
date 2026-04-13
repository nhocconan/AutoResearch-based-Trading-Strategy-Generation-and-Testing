#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1d HTF - Camarilla pivot breakout/mean reversion
    # In bull/bear markets, price tends to respect Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout)
    # Uses 1d Camarilla pivots calculated from prior day, aligned to 6h bars
    # Volume confirmation and ATR filter to avoid chop
    # Target: 50-150 trades over 4 years (12-37/year)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate prior day's Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
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
    
    # Align all HTF indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr_ma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 20-day average
        volume_confirmed = volume[i] > 1.3 * vol_avg_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d[i] > 0.2 * atr_ma_10_aligned[i] if not np.isnan(atr_1d[i]) else False
        
        # Mean reversion at R3/S3 (fade extreme intraday moves)
        mean_rev_long = close[i] <= s3_aligned[i] and close_1d_aligned[i] > s3_aligned[i]
        mean_rev_short = close[i] >= r3_aligned[i] and close_1d_aligned[i] < r3_aligned[i]
        
        # Breakout continuation at R4/S4 (strong momentum)
        breakout_long = close[i] > r4_aligned[i] and high_1d_aligned[i] < r4_aligned[i]
        breakout_short = close[i] < s4_aligned[i] and low_1d_aligned[i] > s4_aligned[i]
        
        # Entry conditions
        enter_long = (mean_rev_long or breakout_long) and volume_confirmed and vol_filter
        enter_short = (mean_rev_short or breakout_short) and volume_confirmed and vol_filter
        
        # Exit conditions: return to pivot or opposite Camarilla level
        exit_long = position == 1 and (close[i] >= pivot_1d_aligned[i] or close[i] <= s3_aligned[i])
        exit_short = position == -1 and (close[i] <= pivot_1d_aligned[i] or close[i] >= r3_aligned[i])
        
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

name = "6h_1d_camarilla_pivot_breakout_meanrev_v1"
timeframe = "6h"
leverage = 1.0