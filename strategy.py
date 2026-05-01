#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 2.0x 20-period volume median.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 2.0x 20-period volume median.
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short) to capture mean reversion.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-50 trades/year on 4h.
# Williams %R is effective in ranging/bear markets for mean reversion, and with trend filter works in bull markets.

name = "4h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Williams %R (14 period) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100,
                          -50)  # fallback to neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Williams %R, EMA50 and volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(highest_high_14[i]) or
            np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Williams %R extreme conditions
        oversold = curr_williams_r < -80.0
        overbought = curr_williams_r > -20.0
        
        # Exit conditions: Williams %R crosses back above -50 (for long) or below -50 (for short)
        exit_long = curr_williams_r > -50.0
        exit_short = curr_williams_r < -50.0
        
        if position == 0:  # Flat - look for new entries
            # Long: Oversold AND uptrend AND volume confirmation
            if oversold and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND downtrend AND volume confirmation
            elif overbought and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Williams %R crosses back above -50 (mean reversion)
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses back below -50 (mean reversion)
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals