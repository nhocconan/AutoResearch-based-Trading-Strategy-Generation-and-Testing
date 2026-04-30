#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (>1.5x average)
# Uses 1d timeframe to reduce trade frequency (target: 30-100 total trades over 4 years)
# 1w EMA50 provides strong trend filter for bull/bear markets
# Volume confirmation >1.5x 20-period average reduces false breakouts
# Discrete position sizing: 0.25 for entries to limit fee drag
# Donchian channels work in all markets, volume confirms legitimacy, trend filter avoids counter-trend

name = "1d_Donchian20_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period) from previous bar to avoid look-ahead
    # Upper = max(high of last 20 bars), Lower = min(low of last 20 bars)
    # Using previous bar's data to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    # Fill first values
    donchian_upper[0] = high[0]
    donchian_lower[0] = low[0]
    
    # Breakout conditions
    breakout_up = close > donchian_upper
    breakout_down = close < donchian_lower
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 50)  # warmup for Donchian (20), volume MA (20), EMA (50)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Donchian upper + above 1w EMA50
                if curr_breakout_up and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Donchian lower + below 1w EMA50
                elif curr_breakout_down and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian lower (reversal) or above Donchian upper (take profit)
            if curr_close < donchian_lower[i] or curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper (reversal) or below Donchian lower (take profit)
            if curr_close > donchian_upper[i] or curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals