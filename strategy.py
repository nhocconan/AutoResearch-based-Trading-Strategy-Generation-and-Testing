#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h EMA trend + volume confirmation
# - Long when price breaks above Donchian(20) high AND 12h EMA(50) > EMA(200) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 12h EMA(50) < EMA(200) AND volume > 1.5x 20-bar avg
# - Exit when price crosses Donchian midpoint (mean reversion structure)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian captures structural breaks; 12h EMA ensures alignment with intermediate trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, midpoint exits work in ranges

name = "4h_12h_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Pre-compute Donchian channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20): 20-period high/low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Donchian breakout conditions
    donchian_breakout_up = close > highest_high
    donchian_breakout_down = close < lowest_low
    
    # Exit when price crosses Donchian midpoint (mean reversion)
    donchian_exit_up = close > donchian_mid  # For shorts
    donchian_exit_down = close < donchian_mid  # For longs
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(donchian_breakout_up[i]) or np.isnan(donchian_breakout_down[i]) or
            np.isnan(donchian_exit_up[i]) or np.isnan(donchian_exit_down[i]) or
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
            # Long when Donchian breakout up AND 12h bullish trend AND volume spike
            if (donchian_breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Donchian breakout down AND 12h bearish trend AND volume spike
            elif (donchian_breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Donchian midpoint
            # Exit when price crosses Donchian midpoint (mean reversion structure)
            if position == 1:  # Long position
                exit_signal = donchian_exit_down[i]  # Price below midpoint
            else:  # Short position
                exit_signal = donchian_exit_up[i]   # Price above midpoint
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals