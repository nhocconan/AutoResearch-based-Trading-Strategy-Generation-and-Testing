#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter
# - Uses 1d Williams %R(14) for mean reversion signals (long below -80, short above -20)
# - Filters by 1w EMA(34) trend: only long when price > EMA, only short when price < EMA
# - Adds volume confirmation: 1d volume > 1.5x 20-period average to avoid low-vol noise
# - Exits when Williams %R crosses back through -50 (mean reversion completion)
# - Position size: 0.25 (25% of capital) for controlled risk
# - Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years) to minimize fee drag
# - Williams %R excels in bear markets by catching oversold bounces and overbought reversals
# - Weekly trend filter prevents trading against the primary trend, reducing whipsaw

name = "1d_1w_williamsr_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Pre-compute HTF indicators (1w)
    close_1w = df_1w['close'].values
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute LTF indicators (1d)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d Volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or close[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with volume confirmation and trend filter
            if (williams_r[i] <= -80 and  # Oversold condition
                volume_spike[i] and       # Volume confirmation
                close[i] > ema_34_1w_aligned[i]):  # Uptrend filter (price above weekly EMA)
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and   # Overbought condition
                  volume_spike[i] and        # Volume confirmation
                  close[i] < ema_34_1w_aligned[i]):  # Downtrend filter (price below weekly EMA)
                position = -1
                signals[i] = -0.25
    
    return signals