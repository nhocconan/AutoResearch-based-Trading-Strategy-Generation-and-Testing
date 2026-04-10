#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d EMA(50) > EMA(200) (bullish trend) AND 12h volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 1d EMA(50) < EMA(200) (bearish trend) AND 12h volume > 1.5x 20-bar avg
# - Exit when price crosses Donchian(20) midline (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian breakouts capture strong momentum; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: trend filter prevents counter-trend trades in bear, breakouts work in bull

name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
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
    
    # Align 1d EMA trend to 12h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute Donchian(20) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Donchian breakout conditions
    breakout_up = close > donchian_high
    breakout_down = close < donchian_low
    # Exit when price crosses midline
    exit_signal = np.abs(close - donchian_mid) < 0.5 * (donchian_high - donchian_low) * 0.1  # Within 10% of midline
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(exit_signal[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when Donchian breakout up AND 1d bullish trend AND volume spike
            if (breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Donchian breakout down AND 1d bearish trend AND volume spike
            elif (breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midline (mean reversion)
            # Exit when price crosses Donchian midline
            exit_condition = exit_signal[i]
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals