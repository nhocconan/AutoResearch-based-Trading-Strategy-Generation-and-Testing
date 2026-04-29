#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above upper Donchian(20) AND price > 1d EMA34 AND volume > 1.5x 24-bar avg
# Short when price breaks below lower Donchian(20) AND price < 1d EMA34 AND volume > 1.5x 24-bar avg
# Exit when price crosses opposite Donchian level (lower for longs, upper for shorts)
# Uses discrete position sizing (0.30) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.
# Donchian channels provide clear breakout structure; 1d EMA34 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h data directly (no HTF needed)
    # We need at least 20 periods for Donchian calculation
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donch = high_series.rolling(window=20, min_periods=20).max().values
    lower_donch = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24)  # Donchian20 warmup + volume MA24 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Donchian levels
        upper_level = upper_donch[i]
        lower_level = lower_donch[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian (mean reversion)
            if curr_close < lower_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian (mean reversion)
            if curr_close > upper_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND price > 1d EMA34 AND volume confirmation
            if curr_close > upper_level and curr_close > ema_34 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below lower Donchian AND price < 1d EMA34 AND volume confirmation
            elif curr_close < lower_level and curr_close < ema_34 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals