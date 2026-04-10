#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with daily trend filter and volume spike
# - Long when Williams %R(14) < -80 (oversold) AND price > daily EMA(50) (uptrend filter) AND volume > 2.0x 20-period volume SMA
# - Short when Williams %R(14) > -20 (overbought) AND price < daily EMA(50) (downtrend filter) AND volume > 2.0x 20-period volume SMA
# - Exit: Williams %R crosses above -50 for longs or below -50 for shorts
# - Position sizing: 0.25 discrete level
# - Williams %R identifies exhaustion points in both bull and bear markets
# - Daily EMA(50) filter ensures we trade with the higher timeframe trend
# - Volume spike requirement reduces false signals and ensures participation
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_williamsr_dailytrend_volumespike_v1"
timeframe = "4h"
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
    
    # Calculate Williams %R(14) on 4h data
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load and align daily EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    for i in range(williams_period, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 2.0x 20-period volume SMA (volume spike)
        vol_spike = volume[i] > 2.0 * volume_sma_20[i]
        
        # Williams %R levels for mean reversion
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        if position == 0:  # Flat - look for entry
            if oversold and close[i] > ema_50_1d_aligned[i] and vol_spike:
                position = 1
                signals[i] = 0.25
            elif overbought and close[i] < ema_50_1d_aligned[i] and vol_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals