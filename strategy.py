#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA50 trend filter and volume confirmation.
# In bull regime (price > 1d EMA50), go long when Williams %R crosses above -80 from oversold with volume spike.
# In bear regime (price < 1d EMA50), go short when Williams %R crosses below -20 from overbought with volume spike.
# Uses 14-period Williams %R for momentum exhaustion signals, 1d EMA50 for regime filter, and 4h volume spike for confirmation.
# Designed for 75-200 total trades over 4 years on BTC/ETH/SOL with discrete sizing to minimize fee drag.

name = "4h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 4h volume regime: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = 0  # previous Williams %R value for crossover detection
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        ema_trend = ema_50_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        prev_wr = prev_williams_r
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(wr) or np.isnan(prev_wr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_williams_r = wr
            continue
            
        # Determine regime: bull if close > 1d EMA50, bear if close < 1d EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R crossover conditions
        crossed_above_oversold = (prev_wr <= -80) and (wr > -80)  # crossing above -80 from below
        crossed_below_overbought = (prev_wr >= -20) and (wr < -20)  # crossing below -20 from above
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Williams %R crosses above -80 (oversold) with volume spike in bull regime
            long_entry = crossed_above_oversold and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Williams %R crosses below -20 (overbought) with volume spike in bear regime
            short_entry = crossed_below_overbought and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on Williams %R crossing below -50 (momentum loss) or regime change to bear
            if wr < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R crossing above -50 (momentum loss) or regime change to bull
            if wr > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        # Update previous Williams %R for next iteration
        prev_williams_r = wr
    
    return signals