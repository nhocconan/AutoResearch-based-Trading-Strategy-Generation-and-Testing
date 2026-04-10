#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# - Long when price breaks above Donchian(20) upper band AND 1d EMA(50) > EMA(200) (bullish trend) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) lower band AND 1d EMA(50) < EMA(200) (bearish trend) AND volume > 1.5x 20-bar avg
# - Exit when price crosses below Donchian(20) middle band (for longs) or above middle band (for shorts)
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Donchian channels provide clear structure; 1d EMA filter avoids counter-trend trades
# - Volume confirmation ensures breakout validity
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull markets via trend-following breakouts; avoids bear markets via 1d EMA filter (only takes longs in bull, shorts in bear)

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1d EMA trend to 4h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute Donchian(20) channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = highest_high
    lower_band = lowest_low
    middle_band = (upper_band + lower_band) / 2.0
    
    # Donchian breakout conditions
    breakout_up = close > upper_band  # Price closes above upper band
    breakout_down = close < lower_band  # Price closes below lower band
    
    # Exit conditions: cross middle band
    exit_long = close < middle_band  # For long positions
    exit_short = close > middle_band  # For short positions
    
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
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when bullish breakout AND 1d bullish trend AND volume spike
            if (breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.30
            # Short when bearish breakout AND 1d bearish trend AND volume spike
            elif (breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at middle band
            # Exit when price crosses middle band
            if position == 1:  # Long position
                if exit_long[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # Short position
                if exit_short[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals