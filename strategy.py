#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d volume spike and 1w trend filter
# - Enter long when Williams %R(14) crosses above -80 (oversold bounce) AND 1d volume > 1.5x 20-period volume SMA AND 1w close > 1w EMA20
# - Enter short when Williams %R(14) crosses below -20 (overbought rejection) AND 1d volume > 1.5x 20-period volume SMA AND 1w close < 1w EMA20
# - Exit: Williams %R crosses below -50 (for longs) or above -50 (for shorts) OR opposite extreme touch
# - Williams %R identifies momentum exhaustion points
# - Volume confirmation ensures institutional participation
# - 1w EMA20 filter avoids counter-trend trades in strong weekly trends
# - Target: 12-30 trades/year to minimize fee drag while capturing high-probability reversals

name = "6h_1d_1w_williamsr_volspike_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute Williams %R for 6h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Pre-compute Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan  # First value has no previous
    cross_above_80 = (williams_r_prev <= -80) & (williams_r > -80)
    cross_below_20 = (williams_r_prev >= -20) & (williams_r < -20)
    cross_below_50 = (williams_r_prev >= -50) & (williams_r < -50)
    cross_above_50 = (williams_r_prev <= -50) & (williams_r > -50)
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute EMA20 for 1w close (trend filter)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 1w close aligned for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(20, n):  # Start after 20-bar warmup for volume SMA and Williams %R
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(williams_r_prev[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close vs EMA20
        uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Williams %R signals
        long_signal = cross_above_80[i] and vol_confirm and uptrend
        short_signal = cross_below_20[i] and vol_confirm and downtrend
        exit_long = cross_below_50[i]  # Exit long when Williams %R crosses below -50
        exit_short = cross_above_50[i]  # Exit short when Williams %R crosses above -50
        
        # Trading logic
        if long_signal:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif short_signal:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Check for exits
            if position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals