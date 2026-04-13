#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike (>2x avg) and chop regime filter
    # Long: price > H3 + volume > 2x 20-period 1d avg + chop < 61.8 (trending)
    # Short: price < L3 + volume > 2x 20-period 1d avg + chop < 61.8 (trending)
    # Exit: price returns to Camarilla Pivot Point (PP)
    # Uses 12h for structure, 1d for volume/chop, no weekly bias to reduce overfitting
    # Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag
    # Volume spike ensures breakout authenticity, chop filter avoids false signals in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.zeros(len(df_12h))
    
    # Get 1d data for volume average and chop (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels on 12h data (using previous bar's OHLC)
    pp_12h = (np.roll(high_12h, 1) + np.roll(low_12h, 1) + np.roll(close_12h, 1)) / 3
    pp_12h[0] = np.nan
    rng_12h = np.roll(high_12h, 1) - np.roll(low_12h, 1)
    h3_12h = pp_12h + (rng_12h * 1.1 / 2)
    l3_12h = pp_12h - (rng_12h * 1.1 / 2)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d data (14-period)
    # True range
    tr = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])),
        np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    )
    tr = np.concatenate([[np.nan], tr])
    # ATR(14) as average true range
    atr = tr
    atr14 = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
    # CHOP = 100 * log10(sum(TR(1)) / (n * ATR(n))) / log10(n)
    sum_tr1 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_denominator = 14 * atr14
    chop = np.where(
        (sum_tr1 > 0) & (chop_denominator > 0),
        np.log10(sum_tr1 / chop_denominator) / np.log10(14) * 100,
        50  # neutral when invalid
    )
    
    # Align all indicators to 12h timeframe
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(h3_12h_aligned[i]) or 
            np.isnan(l3_12h_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2x 20-period 1d average
        volume_confirmed = vol_12h_aligned[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Chop filter: trending market (chop < 61.8)
        is_trending = chop_aligned[i] < 61.8
        
        # Breakout conditions
        breakout_long = (close[i] > h3_12h_aligned[i] and 
                        volume_confirmed and 
                        is_trending)
        breakout_short = (close[i] < l3_12h_aligned[i] and 
                         volume_confirmed and 
                         is_trending)
        
        # Exit conditions: return to Camarilla Pivot Point
        exit_long = position == 1 and close[i] <= pp_12h_aligned[i]
        exit_short = position == -1 and close[i] >= pp_12h_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
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

name = "12h_1d_camarilla_breakout_volume_chop_v2"
timeframe = "12h"
leverage = 1.0