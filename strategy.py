#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (-20/-80 levels)
# Extreme readings (< -90 or > -10) signal exhaustion, not continuation
# Trade reversals from extremes in direction of 1d EMA34 trend (avoid counter-trend)
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Works in bull markets via buying oversold bounces in uptrends and bear markets via selling overbought bounces in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Williams %R calculation (14-period)
        if i >= 14:
            highest_high = np.max(high[i-14:i+1])
            lowest_low = np.min(low[i-14:i+1])
            if highest_high - lowest_low > 0:
                williams_r = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                williams_r = -50  # neutral if no range
        else:
            williams_r = -50
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_spike = False
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: Williams %R < -90 (extreme oversold) AND price above 1d EMA34 (uptrend)
                if williams_r < -90 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -10 (extreme overbought) AND price below 1d EMA34 (downtrend)
                elif williams_r > -10 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought) or price falls below 1d EMA34
            if williams_r > -20 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold) or price rises above 1d EMA34
            if williams_r < -80 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals