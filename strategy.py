#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 12h EMA50 trend filter + volume spike
# - Long when Williams %R(14) < -80 (oversold) and price > EMA50(12h) with volume > 1.5x 20-period average
# - Short when Williams %R(14) > -20 (overbought) and price < EMA50(12h) with volume spike
# - Exit when Williams %R reverts to -50 or opposite extreme triggers
# - Uses discrete sizing (0.25) to minimize fee churn
# - Targets 12-35 trades/year (50-140 total over 4 years) to avoid fee drag
# - Williams %R captures mean reversion in 6h swings while 12h EMA filters counter-trend noise

name = "6h_12h_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h volume confirmation: > 1.5x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Williams %R(14) on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R reverts to -50 or overbought triggers new short
            if williams_r[i] >= -50 or williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R reverts to -50 or oversold triggers new long
            if williams_r[i] <= -50 or williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with trend and volume filters
            if vol_spike_12h_aligned[i]:
                # Long signal: oversold (%R < -80) in 12h uptrend
                if (williams_r[i] < -80 and 
                    prices['close'].iloc[i] > ema_50_12h_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short signal: overbought (%R > -20) in 12h downtrend
                elif (williams_r[i] > -20 and 
                      prices['close'].iloc[i] < ema_50_12h_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals