#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above Camarilla H3 + volume > 1.3x 20-period 1d avg + chop < 61.8 (trending)
    # Short: price breaks below Camarilla L3 + volume > 1.3x 20-period 1d avg + chop < 61.8 (trending)
    # Exit: price returns to Camarilla Pivot Point (PP)
    # Uses 12h timeframe for low frequency, 1d for volume/chop, weekly pivot for structural bias
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    # Works in bull/bear: chop filter avoids false breakouts in ranging markets
    
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
    
    # Get 1d data for volume and chop (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for weekly pivot points (HTF) - structural bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels on 12h data (using previous bar's OHLC)
    # PP = (H+L+C)/3
    # H3 = PP + (H-L)*1.1/2
    # L3 = PP - (H-L)*1.1/2
    # We use previous bar to avoid look-ahead
    pp_12h = (np.roll(high_12h, 1) + np.roll(low_12h, 1) + np.roll(close_12h, 1)) / 3
    pp_12h[0] = np.nan  # first bar has no previous
    rng_12h = np.roll(high_12h, 1) - np.roll(low_12h, 1)
    h3_12h = pp_12h + (rng_12h * 1.1 / 2)
    l3_12h = pp_12h - (rng_12h * 1.1 / 2)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d data (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # Simplified: we use true range and rolling sum
    tr1 = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])),
        np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    )
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr1 = tr1
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    # n * ATR(14) where ATR(14) is average true range over 14 periods
    atr14_avg = pd.Series(atr1).rolling(window=14, min_periods=14).mean().values
    chop_denominator = 14 * atr14_avg
    chop_raw = np.where(
        (atr14 > 0) & (chop_denominator > 0),
        np.log10(atr14 / chop_denominator) / np.log10(14) * 100,
        50  # default to neutral when invalid
    )
    chop = chop_raw
    
    # Weekly bias: bullish if weekly close > weekly PP, bearish if <
    pp_1w = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3
    pp_1w[0] = np.nan
    weekly_bullish = close_1w > pp_1w
    weekly_bearish = close_1w < pp_1w
    
    # Align all indicators to 12h timeframe
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(h3_12h_aligned[i]) or 
            np.isnan(l3_12h_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period 1d average
        # Note: we use 12h volume but compare to 1d average for regime context
        curr_vol_12h = align_htf_to_ltf(prices, df_12h, 
                                       df_12h['volume'].values if 'volume' in df_12h.columns else volume_1d[:len(df_12h)])[i]
        # Fallback: use 1d volume aligned if 12h volume not available in df_12h
        if np.isnan(curr_vol_12h):
            curr_vol_12h = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmed = curr_vol_12h > 1.3 * vol_avg_20_aligned[i]
        
        # Chop filter: trending market (chop < 61.8)
        is_trending = chop_aligned[i] < 61.8
        
        # Weekly pivot direction
        is_weekly_bullish = weekly_bullish_aligned[i] > 0.5
        is_weekly_bearish = weekly_bearish_aligned[i] > 0.5
        
        # Breakout conditions
        breakout_long = (close[i] > h3_12h_aligned[i] and 
                        volume_confirmed and 
                        is_trending and 
                        is_weekly_bullish)
        breakout_short = (close[i] < l3_12h_aligned[i] and 
                         volume_confirmed and 
                         is_trending and 
                         is_weekly_bearish)
        
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

name = "12h_1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0