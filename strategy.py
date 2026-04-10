#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation
# - Long when price breaks above Donchian(20) upper band AND 1d HMA(21) rising (bullish trend) AND 4h volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) lower band AND 1d HMA(21) falling (bearish trend) AND 4h volume > 1.5x 20-bar avg
# - Exit when price closes below Donchian(20) middle (for longs) or above middle (for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian breakout captures strong momentum; 1d HMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_donchian_breakout_hma_volume_v1"
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
    
    # Pre-compute 1d HMA(21) trend: HMA rising/falling
    close_1d = df_1d['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA for half period
    wma_half = np.array([np.nan] * len(close_1d))
    for i in range(half_len - 1, len(close_1d)):
        wma_half[i] = wma(close_1d[i - half_len + 1:i + 1], half_len)
    
    # Calculate WMA for full period
    wma_full = np.array([np.nan] * len(close_1d))
    for i in range(21 - 1, len(close_1d)):
        wma_full[i] = wma(close_1d[i - 21 + 1:i + 1], 21)
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt period
    hma = np.array([np.nan] * len(close_1d))
    for i in range(sqrt_len - 1, len(raw_hma)):
        if not np.isnan(raw_hma[i - sqrt_len + 1:i + 1]).any():
            hma[i] = wma(raw_hma[i - sqrt_len + 1:i + 1], sqrt_len)
    
    # HMA trend: rising if current > previous, falling if current < previous
    hma_rising = np.array([False] * len(hma))
    hma_falling = np.array([False] * len(hma))
    for i in range(1, len(hma)):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-1]):
            hma_rising[i] = hma[i] > hma[i-1]
            hma_falling[i] = hma[i] < hma[i-1]
    
    # Align 1d HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    # Pre-compute Donchian(20) channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper band: highest high over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low over 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian middle band: average of upper and lower
    middle_band = (highest_high + lowest_low) / 2
    
    # Donchian breakout conditions
    breakout_up = close > highest_high  # Price closes above upper band
    breakout_down = close < lowest_low  # Price closes below lower band
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(vol_spike[i]) or np.isnan(middle_band[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 1d HMA rising AND volume spike
            if (breakout_up[i] and 
                hma_rising_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1d HMA falling AND volume spike
            elif (breakout_down[i] and 
                  hma_falling_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Donchian middle
            # Exit when price closes below middle (for longs) or above middle (for shorts)
            if position == 1:  # Long position
                exit_signal = close[i] < middle_band[i]
            else:  # Short position
                exit_signal = close[i] > middle_band[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals