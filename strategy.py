#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d volume spike filter
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1d volume > 1.3x 20-period 1d volume SMA
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1d volume > 1.3x 20-period 1d volume SMA
# - Exit: opposing Williams %R signal or time-based exit (max 12 bars)
# - Uses 1d for volume confirmation, 4h for Williams %R calculation
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to avoid overtrading
# - Williams %R identifies exhaustion points in ranging markets; volume confirmation ensures institutional participation
# - Effective in both bull and bear markets as it captures mean reversion from extremes

name = "4h_1d_williamsr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    
    for i in range(14, n):  # Start from 14 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            bars_since_entry = 0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Get current 1d volume (need to align properly)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        # Volume confirmation: 1d volume > 1.3x 20-period 1d volume SMA
        vol_confirm = vol_1d_aligned[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Williams %R signals
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_cross_above_80 = wr_prev <= -80 and williams_r[i] > -80  # Oversold bounce
        wr_cross_below_20 = wr_prev >= -20 and williams_r[i] < -20   # Overbought rejection
        
        if position == 0:  # Flat - look for entry
            # Require volume confirmation
            if vol_confirm:
                # Long: Williams %R crosses above -80 from oversold
                if wr_cross_above_80:
                    position = 1
                    signals[i] = 0.25
                    bars_since_entry = 0
                # Short: Williams %R crosses below -20 from overbought
                elif wr_cross_below_20:
                    position = -1
                    signals[i] = -0.25
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # In position - look for exit
            bars_since_entry += 1
            
            # Exit conditions: opposing Williams %R signal or max 12 bars
            exit_signal = False
            if position == 1 and wr_cross_below_20:  # Long exit on overbought
                exit_signal = True
            elif position == -1 and wr_cross_above_80:  # Short exit on oversold
                exit_signal = True
            elif bars_since_entry >= 12:  # Time-based exit
                exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals