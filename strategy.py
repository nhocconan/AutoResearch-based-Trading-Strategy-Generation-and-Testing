#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above R4 AND price > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below S4 AND price < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S4 for longs, R4 for shorts)
# Uses discrete position sizing (0.30) to minimize fee churn while capturing moves.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.
# Camarilla levels provide strong support/resistance; 1w EMA34 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "1d_Camarilla_R4S4_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior day's OHLC (1d timeframe)
    # We need the completed prior day's OHLC to calculate today's Camarilla levels
    # Shift by 1 to use only completed prior day
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior day
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior day OHLC to 1d timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 1d bar based on prior day's OHLC
    # Camarilla R4 = Close + (High - Low) * 1.1/2
    # Camarilla S4 = Close - (High - Low) * 1.1/2
    # We use R4/S4 for entries/exits as they are the strongest levels
    range_hl = prior_high_aligned - prior_low_aligned
    r4 = prior_close_aligned + range_hl * 1.1 / 2
    s4 = prior_close_aligned - range_hl * 1.1 / 2
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1) + 1  # EMA34 warmup + 1 for prior day shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1w_aligned[i]
        
        # Camarilla levels
        r4_level = r4[i]
        s4_level = s4[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S4 (mean reversion to median)
            if curr_close < s4_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above R4 (mean reversion to median)
            if curr_close > r4_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above R4 AND price > 1w EMA34 AND volume confirmation
            if curr_close > r4_level and curr_close > ema_34 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S4 AND price < 1w EMA34 AND volume confirmation
            elif curr_close < s4_level and curr_close < ema_34 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals