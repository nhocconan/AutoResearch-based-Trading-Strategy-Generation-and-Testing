#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume regime filter
# - Williams %R(14) on 4h for overbought/oversold conditions
# - Long when %R crosses above -80 from below AND 1d volume > 1.5x 20-period volume SMA (high conviction)
# - Short when %R crosses below -20 from above AND same volume condition
# - Exit: %R crosses back through -50 level (mean reversion completion)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Volume regime filter ensures we trade only during high participation periods
# - Target: 40-80 trades/year on 4h timeframe to balance opportunity and cost

name = "4h_1d_williamsr_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # handle division by zero
    
    # Calculate 1d volume SMA for regime filter
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(30, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime filter: 1d volume > 1.5x 20-period SMA (high participation)
        volume_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_regime = volume_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Williams %R signals (mean reversion from extremes)
        wr_current = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        # Long: %R crosses above -80 from below (oversold bounce)
        long_signal = (wr_prev <= -80) and (wr_current > -80) and vol_regime
        # Short: %R crosses below -20 from above (overbought rejection)
        short_signal = (wr_prev >= -20) and (wr_current < -20) and vol_regime
        # Exit: %R crosses back through -50 (mean reversion completion)
        exit_signal = ((wr_prev <= -50) and (wr_current > -50)) or \
                      ((wr_prev >= -50) and (wr_current < -50))
        
        if position == 0:  # Flat - look for entry
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals