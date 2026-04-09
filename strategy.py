#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Uses Camarilla levels from daily data: L3/S3 for longs, H3/H4 for shorts
# 1d volume confirmation requires current volume > 2.0x 20-period average
# Choppiness regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion
# Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: Camarilla provides intraday mean-reversion levels, volume confirms conviction,
# chop filter avoids false signals in strong trends

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels, volume average, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels for previous day
    rng = prev_high - prev_low
    cam_h3 = prev_close + rng * 1.1 / 4
    cam_h4 = prev_close + rng * 1.1 / 2
    cam_l3 = prev_close - rng * 1.1 / 4
    cam_l4 = prev_close - rng * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (1 bar delay for completed 1d bar)
    cam_h3_12h = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_h4_12h = align_htf_to_ltf(prices, df_1d, cam_h4)
    cam_l3_12h = align_htf_to_ltf(prices, df_1d, cam_l3)
    cam_l4_12h = align_htf_to_ltf(prices, df_1d, cam_l4)
    
    # Calculate 20-period average volume for volume confirmation
    close_1d = pd.Series(df_1d['close'].values)
    vol_1d = pd.Series(df_1d['volume'].values)
    avg_volume_20 = vol_1d.rolling(window=20, min_periods=20).mean().values
    avg_volume_12h = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    # Calculate 14-period Choppiness Index for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    max_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(cam_h3_12h[i]) or np.isnan(cam_h4_12h[i]) or
            np.isnan(cam_l3_12h[i]) or np.isnan(cam_l4_12h[i]) or
            np.isnan(avg_volume_12h[i]) or np.isnan(chop_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-day average volume
        # Need to get the corresponding 12h bar's volume - approximate using daily
        # For simplicity, use daily volume confirmation applied to 12h bars
        vol_confirm = volume[i] > 2.0 * avg_volume_12h[i] if not np.isnan(volume[i]) else False
        
        # Choppiness regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_12h[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3
            if close[i] < cam_l3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3
            if close[i] > cam_h3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume and chop confirmation
            if vol_confirm and chop_filter:
                # Long mean reversion: price touches Camarilla L4 and reverses up
                if low[i] <= cam_l4_12h[i] and close[i] > cam_l4_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short mean reversion: price touches Camarilla H4 and reverses down
                elif high[i] >= cam_h4_12h[i] and close[i] < cam_h4_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals