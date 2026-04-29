#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2x 24-bar avg
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2x 24-bar avg
# Exit when price returns to the Camarilla H3/L3 levels (mean reversion to midpoint)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.
# Camarilla levels provide intraday support/resistance; breakouts indicate institutional participation.
# 1d EMA34 filters counter-trend moves, volume confirmation ensures validity.
# Works in bull markets (buying R3 breakouts) and bear markets (selling S3 breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # We need to get the previous 1d bar's OHLC for each 12h bar
    # Since we're on 12h timeframe, each 12h bar spans half a day
    # Camarilla levels are calculated from daily OHLC, so we use the previous completed 1d bar
    df_1d_prev = df_1d.shift(1)  # Previous completed 1d bar
    high_1d_prev = df_1d_prev['high'].values
    low_1d_prev = df_1d_prev['low'].values
    close_1d_prev = df_1d_prev['close'].values
    
    # Align previous 1d OHLC to 12h timeframe (already aligned since 1d -> 12h is 2:1)
    high_1d_prev_aligned = align_htf_to_ltf(prices, df_1d_prev, high_1d_prev)
    low_1d_prev_aligned = align_htf_to_ltf(prices, df_1d_prev, low_1d_prev)
    close_1d_prev_aligned = align_htf_to_ltf(prices, df_1d_prev, close_1d_prev)
    
    # Calculate Camarilla levels
    # R3 = Close + 1.1 * (High - Low) * 1.1/4
    # S3 = Close - 1.1 * (High - Low) * 1.1/4
    # H3 = Close + 1.1 * (High - Low) * 1.1/6
    # L3 = Close - 1.1 * (High - Low) * 1.1/6
    diff = high_1d_prev_aligned - low_1d_prev_aligned
    r3 = close_1d_prev_aligned + 1.1 * diff * (1.1/4)
    s3 = close_1d_prev_aligned - 1.1 * diff * (1.1/4)
    h3 = close_1d_prev_aligned + 1.1 * diff * (1.1/6)
    l3 = close_1d_prev_aligned - 1.1 * diff * (1.1/6)
    
    # Volume confirmation: >2x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Volume MA warmup and EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Price returns to H3 level (mean reversion)
            if curr_close <= h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3 level (mean reversion)
            if curr_close >= l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1d EMA34 AND volume confirmation
            if curr_close > r3[i] and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND close < 1d EMA34 AND volume confirmation
            elif curr_close < s3[i] and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals