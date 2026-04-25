#!/usr/bin/env python3
"""
6h Volume Spike Reversal with 1w Funding Rate Mean Reversion Filter
Hypothesis: Extreme volume spikes often precede short-term reversals as liquidity gets absorbed. Using 1-week funding rate z-score as a contrarian filter: long when funding is extremely negative (shorts overcrowded), short when extremely positive (longs overcrowded). This combines mean reversion on funding (BTC/ETH edge) with volume exhaustion signals. Works in bull markets (fade longs on high funding + volume spike) and bear markets (fade shorts on negative funding + volume spike). Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for funding rate (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w funding rate z-score (30-week lookback)
    funding = df_1w['funding_rate'].values if 'funding_rate' in df_1w.columns else np.zeros(len(df_1w))
    funding_mean = pd.Series(funding).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding).rolling(window=30, min_periods=30).std().values
    funding_z = (funding - funding_mean) / (funding_std + 1e-8)
    funding_z_aligned = align_htf_to_ltf(prices, df_1w, funding_z)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Price reversal signals: close near recent extremes
    # For longs: price near 10-period low
    # For shorts: price near 10-period high
    period10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    period10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    
    # Normalize price position within recent range (0 = at low, 1 = at high)
    price_range = period10_high - period10_low
    price_range = np.where(price_range == 0, 1, price_range)  # avoid division by zero
    price_position = (close - period10_low) / price_range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(30, 20, 10)  # funding z, vol ma, price range
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(funding_z_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(period10_low[i]) or np.isnan(period10_high[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_vol_spike = vol_spike[i]
        curr_funding_z = funding_z_aligned[i]
        curr_price_pos = price_position[i]
        
        if position == 0:
            # Look for entry signals
            # Long: volume spike + price near low (reversal) + extremely negative funding (contrarian)
            long_entry = (curr_vol_spike and 
                         (curr_price_pos < 0.2) and  # price near 10-period low
                         (curr_funding_z < -2.0))    # funding extremely negative
            
            # Short: volume spike + price near high (reversal) + extremely positive funding (contrarian)
            short_entry = (curr_vol_spike and 
                          (curr_price_pos > 0.8) and  # price near 10-period high
                          (curr_funding_z > 2.0))     # funding extremely positive
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: funding normalizes OR price moves to middle of range OR volume spike in opposite direction
            if (curr_funding_z > -0.5 or  # funding normalized
                curr_price_pos > 0.5 or   # price moved above midpoint
                (curr_vol_spike and curr_price_pos > 0.8)):  # new spike at high = potential failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: funding normalizes OR price moves to middle of range OR volume spike in opposite direction
            if (curr_funding_z < 0.5 or   # funding normalized
                curr_price_pos < 0.5 or   # price moved below midpoint
                (curr_vol_spike and curr_price_pos < 0.2)):  # new spike at low = potential failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolumeSpike_Reversal_1wFundingZ_Contrarian"
timeframe = "6h"
leverage = 1.0