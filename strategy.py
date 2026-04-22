#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume spike
# Williams %R measures overbought/oversold levels: buy when %R crosses above -80 from below, sell when crosses below -20 from above
# Combined with 1d EMA trend filter to ensure we trade with higher timeframe trend
# Volume spike filter ensures we only trade on high conviction moves
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via 1d trend filter - in uptrend we look for oversold bounces, in downtrend for overbought bounces.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R calculation and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # Align Williams %R to 4h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 4h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (oversold bounce) + 1d uptrend + volume spike
            if (williams_r_1d_aligned[i] > -80 and 
                williams_r_1d_aligned[i-1] <= -80 and
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought rejection) + 1d downtrend + volume spike
            elif (williams_r_1d_aligned[i] < -20 and 
                  williams_r_1d_aligned[i-1] >= -20 and
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone (-50) or trend reversal
            if position == 1:
                # Exit on Williams %R crossing below -50 or trend reversal
                if (williams_r_1d_aligned[i] < -50 or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Williams %R crossing above -50 or trend reversal
                if (williams_r_1d_aligned[i] > -50 or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0