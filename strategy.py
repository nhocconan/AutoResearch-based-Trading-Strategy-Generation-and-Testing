#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme + 1d volume regime filter
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1d volume > 1.2x 20-period average
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1d volume > 1.2x 20-period average
# - Exit when Williams %R returns to -50 (mean reversion) or opposite extreme
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets via mean reversion at extremes with volume confirmation
# - Volume filter ensures we trade on genuine participation, not low-volume spikes

name = "4h_1d_williamsr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Williams %R (14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 4h volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma)
    
    # Pre-compute 1d volume regime (20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime = vol_1d > vol_ma_1d  # Above average volume = active market
    
    # Align HTF indicators to 4h timeframe
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(volume_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 (oversold) AND volume confirmation AND 1d active market
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # crossed above -80
                volume_confirm[i] and 
                volume_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 (overbought) AND volume confirmation AND 1d active market
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # crossed below -20
                  volume_confirm[i] and 
                  volume_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R returns to -50 (mean) or opposite extreme with volume
            exit_long = (position == 1 and 
                        (williams_r[i] >= -50 or
                         (williams_r[i] < -80 and volume_confirm[i])))  # re-enter extreme
            exit_short = (position == -1 and 
                         (williams_r[i] <= -50 or
                          (williams_r[i] > -20 and volume_confirm[i])))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals