#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
# - Long when close > Donchian(20) high AND 1d HMA(21) rising AND volume > 1.5x 20-bar avg
# - Short when close < Donchian(20) low AND 1d HMA(21) falling AND volume > 1.5x 20-bar avg
# - Exit when price touches Donchian midpoint OR opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian captures structure; 1d HMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_donchian_hma_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d HMA(21) trend: rising/falling
    close_1d = df_1d['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    # HMA trend: rising if current > previous, falling if current < previous
    hma_rising = np.zeros_like(hma, dtype=bool)
    hma_falling = np.zeros_like(hma, dtype=bool)
    hma_rising[1:] = hma[1:] > hma[:-1]
    hma_falling[1:] = hma[1:] < hma[:-1]
    # Align 1d HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    # Pre-compute Donchian(20) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price > Donchian high AND 1d HMA rising AND volume spike
            if (close[i] > donchian_high[i] and 
                hma_rising_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price < Donchian low AND 1d HMA falling AND volume spike
            elif (close[i] < donchian_low[i] and 
                  hma_falling_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price touches Donchian midpoint OR opposite breakout occurs
            long_exit = (close[i] <= donchian_mid[i]) or (close[i] < donchian_low[i])
            short_exit = (close[i] >= donchian_mid[i]) or (close[i] > donchian_high[i])
            
            if (position == 1 and long_exit) or (position == -1 and short_exit):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals