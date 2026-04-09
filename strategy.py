#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Uses 12h Williams %R(14) for overbought/oversold extremes (more responsive than RSI)
# - Uses 12h EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 6h volume spike (>1.5x 20-period average) for entry confirmation
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets and catches reversals in bear rallies

name = "6h_12h_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 12h Williams %R(14) for overbought/oversold
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_12h) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align 12h Williams %R to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 12h EMA(50) for trend direction
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 6h volume spike confirmation (>1.5x 20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # 6h price data
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if williams_r_aligned[i] > -20:  # Return from oversold
                position = 0
                signals[i] = 0.0
            elif close[i] < ema_50_aligned[i]:  # Price below EMA (trend change)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if williams_r_aligned[i] < -80:  # Return from overbought
                position = 0
                signals[i] = 0.0
            elif close[i] > ema_50_aligned[i]:  # Price above EMA (trend change)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries aligned with 12h trend
            if (williams_r_aligned[i] < -80 and  # Oversold
                close[i] > ema_50_aligned[i] and  # Price above EMA (uptrend)
                volume_spike[i]):  # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (williams_r_aligned[i] > -20 and  # Overbought
                  close[i] < ema_50_aligned[i] and  # Price below EMA (downtrend)
                  volume_spike[i]):  # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals