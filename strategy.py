#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# - Long when price breaks above 4h Donchian upper channel AND 12h close > 12h EMA50 (bullish trend)
# - Short when price breaks below 4h Donchian lower channel AND 12h close < 12h EMA50 (bearish trend)
# - Volume confirmation: 4h volume > 1.8x 20-period volume SMA
# - Exit: opposite Donchian breakout or volume drops below 1.2x volume SMA
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits
# - Works in both bull and bear: trend filter ensures we only trade with 12h momentum,
#   volume confirmation avoids false breakouts, Donchian provides objective entry/exit

name = "4h_12h_donchian_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h close for trend comparison
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate 4h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.8x 20-period volume SMA (strong breakout)
        vol_confirm = volume[i] > 1.8 * volume_sma_20[i]
        
        # Volume exit condition: volume drops below 1.2x SMA (loss of momentum)
        vol_exit = volume[i] < 1.2 * volume_sma_20[i]
        
        # Trend filter: 12h close vs 12h EMA50
        trend_bullish = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_bearish = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Donchian breakout signals (using previous bar's channel)
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper channel
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower channel
        
        # Exit conditions: opposite breakout or loss of volume confirmation
        exit_long = breakout_down or vol_exit
        exit_short = breakout_up or vol_exit
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals