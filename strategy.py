#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (mean reversion)
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h to minimize fee drag.
# Camarilla provides structure from prior day; 1d EMA34 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in both bull (trend continuation) and bear (mean reversion within trend) regimes.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d data
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Extract daily OHLC values for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily OHLC to 12h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Camarilla levels for each 12h bar based on prior day's OHLC
    # We need to use only completed prior day, so shift by 1 (2 bars for 12h timeframe)
    # Since 12h bars: 2 bars per day, shift by 2 to use only completed prior day
    lookback_bars = 2  # 2 * 12h = 1 day
    shifted_high = np.roll(daily_high_aligned, lookback_bars)
    shifted_low = np.roll(daily_low_aligned, lookback_bars)
    shifted_close = np.roll(daily_close_aligned, lookback_bars)
    # Set first values to NaN as we don't have prior completed day
    shifted_high[:lookback_bars] = np.nan
    shifted_low[:lookback_bars] = np.nan
    shifted_close[:lookback_bars] = np.nan
    
    # Calculate Camarilla levels
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    # where C, H, L are from prior completed day
    hl_range = shifted_high - shifted_low
    r3 = shifted_close + hl_range * 1.1 / 2.0
    s3 = shifted_close - hl_range * 1.1 / 2.0
    # Also calculate R4 and S4 for exit levels (more extreme)
    r4 = shifted_close + hl_range * 1.1
    s4 = shifted_close - hl_range * 1.1
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + lookback_bars  # EMA34 and volume warmup + lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Camarilla levels
        r3_level = r3[i]
        s3_level = s3[i]
        r4_level = r4[i]
        s4_level = s4[i]
        
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
            # Long when price breaks above R3 AND price > 1d EMA34 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 1d EMA34 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals