#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels provide robust breakout levels - breakouts indicate strong momentum shifts
# 1w EMA50 provides long-term trend filter to avoid counter-trend trades in bear markets
# Volume confirmation (>1.5x average) ensures breakout legitimacy with lower frequency
# Works in bull/bear: breakouts occur in all regimes, volume confirms legitimacy, trend filter reduces false signals
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_Donchian20_1wEMA50_Trend_Volume_v1"
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
    
    # Calculate 1d Donchian channels (20-period) from previous bar
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Need previous bar's levels to avoid look-ahead
    donchian_upper_prev = np.roll(donchian_upper, 1)
    donchian_lower_prev = np.roll(donchian_lower, 1)
    donchian_upper_prev[0] = np.nan
    donchian_lower_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > donchian_upper_prev
    breakout_down = close < donchian_lower_prev
    
    # Volume confirmation: volume > 1.5x 20-period average (strict to reduce trades)
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
    
    start_idx = max(100, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper_prev[i]) or 
            np.isnan(donchian_lower_prev[i]) or
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
            if curr_close < donchian_lower_prev[i] or curr_close > donchian_upper_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper (reversal) or below Donchian lower (take profit)
            if curr_close > donchian_upper_prev[i] or curr_close < donchian_lower_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals