#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout for direction and 1h for precise timing.
- Primary timeframe: 1h for entry timing (reduces whipsaw vs pure 4h signals).
- HTF: 4h Donchian(20) for trend structure (bullish if price > upper band, bearish if price < lower band).
- Volume: Current 1h volume > 1.5 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above 4h Donchian high (aligned) AND volume spike.
         Short when price breaks below 4h Donchian low (aligned) AND volume spike.
- Exit: Opposite Donchian breakout (aligned) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and fee churn.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h (waits for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Calculate 20-period volume MA on 1h
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need enough bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session or data not ready
        if not in_session[i] or \
           np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_donchian = donchian_high_aligned[i]
        lower_donchian = donchian_low_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if volume_spike[i]:
                # Bullish breakout: price breaks above upper Donchian
                if curr_high > upper_donchian:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below lower Donchian
                elif curr_low < lower_donchian:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR loss of volume confirmation
            if curr_low < lower_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR loss of volume confirmation
            if curr_high > upper_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hDirection_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0