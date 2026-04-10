#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w EMA(50) > EMA(200) AND volume > 1.5x 20-day avg
# - Short when price breaks below Donchian(20) low AND 1w EMA(50) < EMA(200) AND volume > 1.5x 20-day avg
# - Exit when price crosses the Donchian(20) midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian breakouts capture momentum; 1w EMA filter ensures alignment with weekly trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion exit prevents overstay

name = "1d_1w_donchian_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(50) vs EMA(200)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1w EMA trend to 1d timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish)
    
    # Pre-compute Donchian(20) channels on 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Donchian breakout conditions
    breakout_up = close > highest_high
    breakout_down = close < lowest_low
    
    # Exit when price crosses Donchian midpoint
    exit_signal = (close > donchian_mid) & (np.roll(close, 1) <= donchian_mid) | \
                  (close < donchian_mid) & (np.roll(close, 1) >= donchian_mid)
    # Handle first bar
    exit_signal[0] = False
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
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
            # Long when breakout up AND 1w bullish trend AND volume spike
            if (breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when breakout down AND 1w bearish trend AND volume spike
            elif (breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Donchian midpoint
            # Exit when price crosses Donchian midpoint (mean reversion)
            if exit_signal[i]:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals