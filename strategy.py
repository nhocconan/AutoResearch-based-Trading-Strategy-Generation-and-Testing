#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakout captures institutional breakout moves. 1w EMA50 filter ensures alignment with higher timeframe trend.
# Volume confirmation ensures breakout validity. Works in bull (breakouts above upper band) and bear (breakdowns below lower band).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA50 on weekly data for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Close above upper Donchian band AND price above weekly EMA50 AND volume spike
            if close_val > high_20_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Donchian band AND price below weekly EMA50 AND volume spike
            elif close_val < low_20_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below weekly EMA50 (trend change) or at lower Donchian band (mean reversion)
            if close_val < ema_val or close_val < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above weekly EMA50 (trend change) or at upper Donchian band (mean reversion)
            if close_val > ema_val or close_val > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals