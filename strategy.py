#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Elder Ray (Bull/Bear Power) + volume confirmation
# - Williams %R(14) on 6h: oversold < -80 for long, overbought > -20 for short
# - 1d Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Williams %R < -80 AND Bull Power > 0 AND volume > 1.5x 20-period average
# - Short when Williams %R > -20 AND Bear Power > 0 AND volume > 1.5x 20-period average
# - Exit when Williams %R crosses back through -50 (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R captures momentum extremes, Elder Ray confirms trend strength
# - Volume confirmation reduces false signals

name = "6h_1d_williams_elder_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13  # Bull Power = High - EMA(13)
    bear_power = ema_13 - low_1d   # Bear Power = EMA(13) - Low
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND Bull Power positive AND volume spike
            if (williams_r[i] < -80 and 
                bull_power_aligned[i] > 0 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND Bear Power positive AND volume spike
            elif (williams_r[i] > -20 and 
                  bear_power_aligned[i] > 0 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses back through -50 (mean reversion)
            exit_long = (position == 1 and williams_r[i] > -50)
            exit_short = (position == -1 and williams_r[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals