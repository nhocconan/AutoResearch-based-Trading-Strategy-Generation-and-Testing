#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1w EMA34 to filter trend direction (avoid counter-trend trades in ranging/choppy markets)
# Donchian(20) breakout captures strong momentum moves with clear structure
# Volume > 1.5x 20-period average confirms institutional participation
# Discrete position sizing (0.25) with opposite Donchian(10) exit for risk control
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag
# Works in bull/bear via 1w EMA34 trend filter - only trades in direction of weekly trend
# Designed for BTC/ETH focus with proven Donchian structure + volume confirmation edge

name = "1d_Donchian20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    # Donchian upper = max(high, lookback)
    # Donchian lower = min(low, lookback)
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_roll_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_roll_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Donchian(20) and 1w EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll_20[i]) or np.isnan(low_roll_20[i]) or 
            np.isnan(high_roll_10[i]) or np.isnan(low_roll_10[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high_20 = high_roll_20[i]
        curr_donchian_low_20 = low_roll_20[i]
        curr_donchian_high_10 = high_roll_10[i]
        curr_donchian_low_10 = low_roll_10[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Donchian(10) lower band (tight stop/profit)
            if curr_close < curr_donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Donchian(10) upper band (tight stop/profit)
            if curr_close > curr_donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above Donchian(20) upper band with 1w EMA34 uptrend and volume confirmation
            if curr_close > curr_donchian_high_20 and curr_close > curr_ema34_1w and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian(20) lower band with 1w EMA34 downtrend and volume confirmation
            elif curr_close < curr_donchian_low_20 and curr_close < curr_ema34_1w and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals