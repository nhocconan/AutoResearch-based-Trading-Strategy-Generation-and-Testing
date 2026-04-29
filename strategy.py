#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R1 AND price > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when price breaks below S1 AND price < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when price crosses opposite Camarilla level (S1 for longs, R1 for shorts)
# Uses discrete position sizing (0.30) to balance capture and fee drag.
# Target: 100-200 total trades over 4 years (25-50/year) on 4h.
# Camarilla levels provide mathematical support/resistance; 1d EMA34 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).
# Proven pattern from DB: 4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeConfirm_v1 achieved test Sharpe 0.495 with 199 trades.
# This version tightens volume threshold from 2.0x to 1.8x to increase trade frequency while maintaining edge.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Extract prior day's OHLC (1d timeframe) for Camarilla levels
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior day
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior day OHLC to 4h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 4h bar based on prior day's OHLC
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    range_hl = prior_high_aligned - prior_low_aligned
    r1 = prior_close_aligned + range_hl * 1.1 / 12
    s1 = prior_close_aligned - range_hl * 1.1 / 12
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1) + 1  # EMA34 warmup + 1 for prior day shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Camarilla levels
        r1_level = r1[i]
        s1_level = s1[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S1 (mean reversion to median)
            if curr_close < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above R1 (mean reversion to median)
            if curr_close > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND price > 1d EMA34 AND volume confirmation
            if curr_close > r1_level and curr_close > ema_34 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S1 AND price < 1d EMA34 AND volume confirmation
            elif curr_close < s1_level and curr_close < ema_34 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals