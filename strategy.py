#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop filter
    # Uses 1d Camarilla levels (proven edge) with 4h precision entry
    # Volume confirmation avoids low-quality breakouts
    # Chop regime filter (EHLERS) avoids whipsaws in ranging markets
    # Target: 75-200 trades over 4 years (19-50/year) for optimal fee drag balance
    
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
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # We use the prior day's range to calculate today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d EHLERS chop filter (34-period)
    def hl2(high, low):
        return (high + low) / 2
    
    def true_range(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    hl2_1d = hl2(high_1d, low_1d)
    tr_1d = true_range(high_1d, low_1d, close_1d)
    
    # EHLERS chop: sum of TR / (sum of HL2 changes)
    sum_tr = pd.Series(tr_1d).rolling(window=34, min_periods=34).sum().values
    hl2_diff = np.abs(np.diff(hl2_1d, prepend=hl2_1d[0]))
    sum_hl2_diff = pd.Series(hl2_diff).rolling(window=34, min_periods=34).sum().values
    chop = 100 * np.log10(sum_tr / sum_hl2_diff) / np.log10(34)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Chop filter: avoid extreme ranging (chop > 61.8) or extreme trending (chop < 38.2)
        # We prefer moderate chop (38.2 <= chop <= 61.8) for breakout reliability
        chop_filter = (chop_aligned[i] >= 38.2) and (chop_aligned[i] <= 61.8)
        
        # Breakout conditions using Camarilla H4/L4 levels
        breakout_up = close[i] > camarilla_h4_aligned[i]
        breakout_down = close[i] < camarilla_l4_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and chop_filter
        enter_short = breakout_down and volume_confirmed and chop_filter
        
        # Exit conditions: price returns to prior day's close (mean reversion)
        exit_long = position == 1 and close[i] <= prev_close_1d[i]
        exit_short = position == -1 and close[i] >= prev_close_1d[i]
        
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

name = "4h_1d_camarilla_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0