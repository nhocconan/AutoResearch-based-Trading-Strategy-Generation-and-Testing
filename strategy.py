#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Uses 1w EMA(34) for primary trend direction (long when price > EMA34, short when price < EMA34)
# - Uses 6h Williams %R(14) for overbought/oversold signals (long when %R crosses above -80, short when crosses below -20)
# - Requires 6h volume > 1.5x 20-period average volume for confirmation
# - Only takes mean reversion trades aligned with 1w trend (long in uptrend, short in downtrend)
# - Target: 12-35 trades/year on 6h timeframe (50-140 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets which appear in both bull and bear phases
# - 1w trend filter prevents trading against the primary trend during strong moves

name = "6h_1w_williamsr_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Pre-compute session-independent (6h bars less session-sensitive)
    
    # 1w EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = np.where(close_1w > ema_34_1w, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Align 1w trend to 6h
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # 6h Williams %R(14) for mean reversion signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, where -20 is overbought, -80 is oversold
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, ((highest_high - close) / rr) * -100, -50)
    
    # 6h volume confirmation
    volume = prices['volume'].values
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(avg_volume > 0, volume / avg_volume, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(35, n):  # Start after warmup period for 1w EMA34
        # Skip if any required data is invalid
        if (np.isnan(trend_1w_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_ratio[i]) or
            np.isnan(avg_volume[i]) or
            avg_volume[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion completion or trend change
            if williams_r[i] >= -20:  # Return from oversold
                position = 0
                signals[i] = 0.0
            elif trend_1w_aligned[i] == -1:  # 1w trend turned down
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion completion or trend change
            if williams_r[i] <= -80:  # Return from overbought
                position = 0
                signals[i] = 0.0
            elif trend_1w_aligned[i] == 1:  # 1w trend turned up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries aligned with 1w trend
            if (williams_r[i] <= -80 and  # Oversold
                volume_ratio[i] > 1.5 and  # Volume confirmation
                trend_1w_aligned[i] == 1):  # 1w uptrend
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and  # Overbought
                  volume_ratio[i] > 1.5 and  # Volume confirmation
                  trend_1w_aligned[i] == -1):  # 1w downtrend
                position = -1
                signals[i] = -0.25
    
    return signals