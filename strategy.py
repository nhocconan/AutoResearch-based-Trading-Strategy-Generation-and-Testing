# USDC/USDT Parity Mean Reversion with Bollinger Bands
# Strategy exploits temporary deviations from 1:1 parity between USDC and USDT
# Uses Bollinger Bands on the price ratio to identify mean reversion opportunities
# Works in both bull and bear markets as it's market-neutral
# Target: Low frequency, high win rate trades with minimal drawdown

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate deviation from 1.0 parity
    # For stablecoin pairs, we expect price to oscillate around 1.0
    deviation = close - 1.0
    
    # Bollinger Bands on deviation (20-period, 2 standard deviations)
    dev_series = pd.Series(deviation)
    bb_middle = dev_series.rolling(window=20, min_periods=20).mean().values
    bb_std = dev_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume filter: avoid low liquidity periods
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long (price > 1), -1: short (price < 1)
    size = 0.25   # 25% position size
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        dev = deviation[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require sufficient liquidity
        vol_filter = vol_now > 0.5 * vol_avg  # Low threshold for stablecoins
        
        if position == 0:
            # Enter long when price significantly below 1.0 (undervalued USDC)
            if dev < bb_lower[i] and vol_filter:
                signals[i] = size
                position = 1
            # Enter short when price significantly above 1.0 (overvalued USDC)
            elif dev > bb_upper[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price returns to mean (1.0) or crosses above
            if dev >= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to mean (1.0) or crosses below
            if dev <= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "USDC_USDT_Pairs_MeanReversion_BB20_2"
timeframe = "1h"  # 1h provides good balance for mean reversion
leverage = 1.0