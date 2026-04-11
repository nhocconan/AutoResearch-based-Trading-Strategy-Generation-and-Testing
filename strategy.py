#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d volume spike and 1w trend filter
# - Enter long when Williams %R(14) crosses above -80 (oversold) AND 1d volume > 2.0x 20-period volume SMA AND 1w close > 1w EMA20
# - Enter short when Williams %R(14) crosses below -20 (overbought) AND 1d volume > 2.0x 20-period volume SMA AND 1w close < 1w EMA20
# - Exit: Williams %R crosses below -50 for longs or above -50 for shorts
# - Williams %R identifies extreme price levels likely to reverse
# - Volume confirmation ensures reversal validity (institutional participation)
# - 1w EMA20 filter aligns with higher timeframe trend to avoid counter-trend trades
# - Target: 20-35 trades/year to minimize fee drag while capturing high-probability reversals

name = "4h_1d_1w_williamsr_reversal_v2"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for Williams %R and volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1d OHLC for Williams %R calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R for 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_14 = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r_14 = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_14)
    
    # Align Williams %R to 4h timeframe (using completed 1d bar)
    williams_r_14_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14)
    
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
    
    for i in range(30, n):  # Start after 30-bar warmup for 14-period Williams %R and 20-period volume SMA
        # Skip if any required data is invalid
        if (np.isnan(williams_r_14_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close vs EMA20
        uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Williams %R reversal signals
        williams_r = williams_r_14_aligned[i]
        williams_r_prev = williams_r_14_aligned[i-1] if i > 0 else williams_r
        
        # Long signal: Williams %R crosses above -80 from below (oversold reversal)
        long_reversal = (williams_r > -80) and (williams_r_prev <= -80)
        # Short signal: Williams %R crosses below -20 from above (overbought reversal)
        short_reversal = (williams_r < -20) and (williams_r_prev >= -20)
        
        # Exit signals: Williams %R crosses -50 level
        long_exit = (williams_r < -50) and (williams_r_prev >= -50)  # Exit long when momentum fades
        short_exit = (williams_r > -50) and (williams_r_prev <= -50)  # Exit short when momentum fades
        
        # Trading logic
        if long_reversal and vol_confirm and uptrend:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif short_reversal and vol_confirm and downtrend:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Check for mean reversion exits
            if position == 1 and long_exit:
                position = 0
                signals[i] = 0.0
            elif position == -1 and short_exit:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals