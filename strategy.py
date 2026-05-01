#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversals with 1d EMA50 trend filter and volume confirmation.
# Williams %R < -80 = oversold (long setup), > -20 = overbought (short setup).
# Enter long when %R crosses above -80 from below AND price > 1d EMA50 (uptrend) AND volume > 1.3x 20-period volume median.
# Enter short when %R crosses below -20 from above AND price < 1d EMA50 (downtrend) AND volume > 1.3x 20-period volume median.
# Exit on opposite %R extreme (%R > -50 for longs, %R < -50 for shorts) to reduce whipsaw.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
# Works in bull (buy oversold dips with trend) and bear (sell overbought rallies with trend).

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams %R (14-period) on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range=0
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Williams %R, EMA50, and volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.3)
        
        # Williams %R conditions
        oversold = curr_williams_r < -80
        overbought = curr_williams_r > -20
        # Cross above -80 (bullish reversal)
        cross_above_80 = (prev_williams_r <= -80) and (curr_williams_r > -80)
        # Cross below -20 (bearish reversal)
        cross_below_20 = (prev_williams_r >= -20) and (curr_williams_r < -20)
        # Exit conditions: %R > -50 for longs, %R < -50 for shorts
        exit_long = curr_williams_r > -50
        exit_short = curr_williams_r < -50
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R cross above -80 AND uptrend AND volume confirmation
            if cross_above_80 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R cross below -20 AND downtrend AND volume confirmation
            elif cross_below_20 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R > -50 (momentum fading)
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R < -50 (momentum fading)
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals