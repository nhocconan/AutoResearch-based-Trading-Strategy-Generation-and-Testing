#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 12h EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 12h EMA34), go long when Williams %R crosses above -80 (oversold bounce) with volume spike.
# In bear regime (price < 12h EMA34), go short when Williams %R crosses below -20 (overbought rejection) with volume spike.
# Uses Williams %R for mean-reversion entries within the trend, 12h EMA34 for regime filter, and 6h volume spike for confirmation.
# Designed for 50-150 total trades over 4 years (12-37/year) on BTC/ETH.

name = "6h_WilliamsR_12hEMA34_VolumeSpike_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R (14) and EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate Williams %R (14) on 12h
    highest_high = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low)
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h (wait for 12h bar to complete)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 12h EMA34 for trend filter
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate volume spike: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        wr = williams_r_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 12h EMA34, bear if close < 12h EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R crossovers with previous bar (need i-1 for crossover)
        if i > 0:
            wr_prev = williams_r_aligned[i-1]
            wr_cross_above_80 = (wr_prev <= -80) and (wr > -80)  # Oversold bounce
            wr_cross_below_20 = (wr_prev >= -20) and (wr < -20)  # Overbought rejection
        else:
            wr_cross_above_80 = False
            wr_cross_below_20 = False
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Williams %R crosses above -80 (oversold bounce) with volume spike
            long_entry = wr_cross_above_80 and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Williams %R crosses below -20 (overbought rejection) with volume spike
            short_entry = wr_cross_below_20 and vol_spike
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
            # Exit on Williams %R cross below -50 (momentum loss) or regime change to bear
            if wr < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R cross above -50 (momentum loss) or regime change to bull
            if wr > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals