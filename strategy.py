#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA(50/200) trend + volume confirmation
# - Long when price breaks above Donchian upper(20) AND 12h EMA(50) > EMA(200) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian lower(20) AND 12h EMA(50) < EMA(200) AND volume > 1.5x 20-bar avg
# - Exit when price touches Donchian midpoint (mean reversion) OR opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian captures structural breaks; 12h EMA filter ensures alignment with intermediate trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_12h_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA trend filter: EMA(50) vs EMA(200)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 12h EMA trend to 4h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_bearish)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Donchian breakout conditions
    breakout_up = close > donchian_upper
    breakout_down = close < donchian_lower
    
    # Exit conditions: price touches midpoint or opposite breakout
    exit_long = close <= donchian_mid
    exit_short = close >= donchian_mid
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(exit_long[i]) or np.isnan(exit_short[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when bullish breakout AND 12h bullish trend AND volume spike
            if (breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when bearish breakout AND 12h bearish trend AND volume spike
            elif (breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price touches midpoint OR opposite breakout with volume/spike
            exit_condition = False
            if position == 1:  # Long position
                if exit_long[i]:
                    exit_condition = True
                elif (breakout_down[i] and vol_spike[i]):  # Strong opposite breakout
                    exit_condition = True
            else:  # Short position
                if exit_short[i]:
                    exit_condition = True
                elif (breakout_up[i] and vol_spike[i]):  # Strong opposite breakout
                    exit_condition = True
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals