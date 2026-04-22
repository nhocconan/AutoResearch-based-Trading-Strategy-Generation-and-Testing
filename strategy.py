#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h EMA50 trend filter and volume spike
# Williams %R identifies overbought/oversold conditions: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) and rising + 12h uptrend + volume spike
# Short when %R > -20 (overbought) and falling + 12h downtrend + volume spike
# Combines momentum oscillator with trend filter for better timing
# Designed for 4h timeframe to target 20-40 trades/year per symbol.
# Works in bull (catches bounces from oversold) and bear (sells rallies from overbought)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for higher timeframe trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R slope (momentum)
    williams_r_slope = williams_r - np.roll(williams_r, 1)
    williams_r_slope[0] = 0
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r_slope[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) and rising + 12h uptrend + volume spike
            if (williams_r[i] < -80 and 
                williams_r_slope[i] > 0 and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) and falling + 12h downtrend + volume spike
            elif (williams_r[i] > -20 and 
                  williams_r_slope[i] < 0 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50 or trend reversal
            if position == 1:
                # Exit on Williams %R >= -50 or trend reversal
                if (williams_r[i] >= -50 or 
                    close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Williams %R <= -50 or trend reversal
                if (williams_r[i] <= -50 or 
                    close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0