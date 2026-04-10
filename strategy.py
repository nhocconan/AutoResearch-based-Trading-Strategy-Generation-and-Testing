#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation
# - Long when price breaks above Donchian(20) upper band AND 1w HMA(21) is rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) lower band AND 1w HMA(21) is falling AND volume > 1.5x 20-bar avg
# - Exit when price crosses opposite Donchian band or volume drops below average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian captures structural breaks; 1w HMA ensures alignment with weekly trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_donchian_breakout_hma_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 1w HMA(21) trend: rising/falling
    close_1w = df_1w['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # Hull Moving Average calculation
    wma_half = pd.Series(close_1w).ewm(span=half_length, adjust=False).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_length, adjust=False).mean().values
    
    # HMA trend: rising if current > previous, falling if current < previous
    hma_rising = hma > np.concatenate([[hma[0]], hma[:-1]])
    hma_falling = hma < np.concatenate([[hma[0]], hma[:-1]])
    
    # Align 1w HMA trend to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling)
    
    # Pre-compute Donchian(20) channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
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
            # Long when price breaks above Donchian upper AND 1w HMA rising AND volume spike
            if (close[i] > highest_high[i] and 
                hma_rising_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1w HMA falling AND volume spike
            elif (close[i] < lowest_low[i] and 
                  hma_falling_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses opposite Donchian band
            exit_long = position == 1 and close[i] < lowest_low[i]
            exit_short = position == -1 and close[i] > highest_high[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals