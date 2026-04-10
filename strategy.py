#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA(50) (uptrend) AND 6h volume > 1.3x 20-period 6h volume SMA
# - Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA(50) (downtrend) AND 6h volume > 1.3x 20-period 6h volume SMA
# - Exit: Williams %R crosses above -50 for longs or below -50 for shorts
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R identifies exhaustion points; 1d EMA filter ensures trading with higher timeframe trend
# - Volume confirmation reduces false signals in low-participation moves

name = "6h_1d_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) on 6h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 20-period volume SMA for 6h confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    for i in range(50, n):  # Start after Williams %R and EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        if position == 0:  # Flat - look for entry
            # Long: oversold + uptrend (price > 1d EMA50) + volume
            if oversold and close[i] > ema_50_1d_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: overbought + downtrend (price < 1d EMA50) + volume
            elif overbought and close[i] < ema_50_1d_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when Williams %R crosses above -50 (mean reversion)
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when Williams %R crosses below -50 (mean reversion)
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals