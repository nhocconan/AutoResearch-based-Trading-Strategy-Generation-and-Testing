#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation
# - Williams %R: momentum oscillator identifying overbought/oversold conditions
# - Long when %R crosses above -80 (oversold recovery) in 1d uptrend with volume spike
# - Short when %R crosses below -20 (overbought breakdown) in 1d downtrend with volume spike
# - Uses 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
# - 1d EMA50 filter ensures trading with higher timeframe trend direction
# - 4h volume > 1.5x 20-period average confirms breakout strength
# - Discrete position sizing (0.25) to minimize fee churn
# - Williams %R exit: opposite threshold crossover (-20 for long, -80 for short)

name = "4h_1d_williamsr_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 4h Williams %R (14-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_4h) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Williams %R signals
    williams_r_oversold = williams_r < -80  # oversold condition
    williams_r_overbought = williams_r > -20  # overbought condition
    williams_r_cross_above_oversold = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_r_cross_below_overbought = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -20 (overbought)
            if williams_r_cross_below_overbought[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -80 (oversold)
            if williams_r_cross_above_oversold[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R signals with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: Williams %R crosses above -80 in 1d uptrend
                if williams_r_cross_above_oversold[i] and close_4h[i] > ema_50_1d_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short signal: Williams %R crosses below -20 in 1d downtrend
                elif williams_r_cross_below_overbought[i] and close_4h[i] < ema_50_1d_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals