#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA50 trend filter
# Long when price breaks above Donchian upper (20-period) AND volume > 1.5x 20-period average AND price > 12h EMA50
# Short when price breaks below Donchian lower (20-period) AND volume > 1.5x 20-period average AND price < 12h EMA50
# Exit when price crosses Donchian midpoint OR trend reverses
# Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Donchian provides objective breakout levels, volume confirms conviction, 12h EMA filters counter-trend noise

name = "4h_Donchian20_Volume_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (completed 12h bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period) - using 4h data
    period = 20
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20, 50)  # warmup for Donchian, volume MA, and EMA50
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_dc_middle = donchian_middle[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above Donchian upper AND price > 12h EMA50 (bullish trend)
                if curr_high > curr_dc_upper and curr_close > curr_ema50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below Donchian lower AND price < 12h EMA50 (bearish trend)
                elif curr_low < curr_dc_lower and curr_close < curr_ema50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below Donchian middle OR trend turns bearish
            if curr_close < curr_dc_middle or curr_close < curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above Donchian middle OR trend turns bullish
            if curr_close > curr_dc_middle or curr_close > curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals