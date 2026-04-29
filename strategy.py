#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 75-150 total trades over 4 years (19-37/year) on 4h.
# Camarilla R3/S3 are strong intraday levels with good breakout reliability.
# 12h EMA50 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 12h data for Camarilla pivot levels (using prior bar's OHLC)
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Extract prior bar's OHLC (12h timeframe)
    # We need the completed prior bar's OHLC to calculate current bar's Camarilla levels
    # Shift by 1 to use only completed prior bar
    prior_high = np.roll(df_12h['high'].values, 1)
    prior_low = np.roll(df_12h['low'].values, 1)
    prior_close = np.roll(df_12h['close'].values, 1)
    # Set first value to NaN as we don't have prior bar
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior bar OHLC to 4h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_12h, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_12h, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_12h, prior_close)
    
    # Calculate Camarilla levels for each 4h bar based on prior bar's OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    # We use R3/S3 for entries/exits as they are strong intraday levels
    range_hl = prior_high_aligned - prior_low_aligned
    r3 = prior_close_aligned + range_hl * 1.1 / 4
    s3 = prior_close_aligned - range_hl * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_12h_aligned[i]
        
        # Camarilla levels
        r3_level = r3[i]
        s3_level = s3[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion to median)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion to median)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 12h EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 12h EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals