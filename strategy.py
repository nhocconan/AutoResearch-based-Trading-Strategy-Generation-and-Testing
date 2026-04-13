#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla breakout with 1d volume spike and chop regime filter
    # Long: price breaks above Camarilla H3 + volume > 1.5x 20-period 1d avg + chop < 61.8 (trending)
    # Short: price breaks below Camarilla L3 + volume > 1.5x 20-period 1d avg + chop < 61.8 (trending)
    # Exit: price returns to Camarilla Pivot Point (PP)
    # Uses 4h timeframe for primary signals, 1d for volume/chop confirmation
    # Target: 75-150 total trades over 4 years (19-37/year) to minimize fee drag
    # Volume spike + chop filter reduces false breakouts in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else None
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values if 'volume' in df_4h.columns else None
    
    # Get 1d data for volume and chop (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels on 4h data (using previous bar's OHLC)
    # PP = (H+L+C)/3
    # H3 = PP + (H-L)*1.1/2
    # L3 = PP - (H-L)*1.1/2
    pp_4h = (np.roll(high_4h, 1) + np.roll(low_4h, 1) + np.roll(close_4h, 1)) / 3
    pp_4h[0] = np.nan  # first bar has no previous
    rng_4h = np.roll(high_4h, 1) - np.roll(low_4h, 1)
    h3_4h = pp_4h + (rng_4h * 1.1 / 2)
    l3_4h = pp_4h - (rng_4h * 1.1 / 2)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d data (14-period)
    # True Range
    tr1 = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])),
        np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    )
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    
    # ATR(14) = average of TR over 14 periods
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 1 period (just TR itself)
    sum_tr1 = tr1
    
    # CHOP = 100 * log10(sum(TR(1)) / (n * ATR(n))) / log10(n)
    chop_denominator = 14 * atr14
    chop_raw = np.where(
        (sum_tr1 > 0) & (chop_denominator > 0),
        np.log10(sum_tr1 / chop_denominator) / np.log10(14) * 100,
        50  # default to neutral when invalid
    )
    chop = chop_raw
    
    # Align all indicators to 4h timeframe
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align 4h volume if available, otherwise use 1d volume aligned
    if vol_4h is not None:
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    else:
        vol_4h_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period 1d average
        volume_confirmed = vol_4h_aligned[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Chop filter: trending market (chop < 61.8)
        is_trending = chop_aligned[i] < 61.8
        
        # Breakout conditions
        breakout_long = (close[i] > h3_4h_aligned[i] and 
                        volume_confirmed and 
                        is_trending)
        breakout_short = (close[i] < l3_4h_aligned[i] and 
                         volume_confirmed and 
                         is_trending)
        
        # Exit conditions: return to Camarilla Pivot Point
        exit_long = position == 1 and close[i] <= pp_4h_aligned[i]
        exit_short = position == -1 and close[i] >= pp_4h_aligned[i]
        
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

name = "4h_1d_camarilla_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0