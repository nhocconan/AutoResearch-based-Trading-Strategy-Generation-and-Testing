#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation.
# In bull regime (price > 1d EMA34), go long when Williams %R crosses above -80 (oversold bounce) with volume spike.
# In bear regime (price < 1d EMA34), go short when Williams %R crosses below -20 (overbought bounce) with volume spike.
# Uses Williams %R from 12h data, 1d EMA34 for regime filter, and 12h volume spike for confirmation.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.
# Focus on BTC/ETH as primary symbols.

name = "12h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for Williams %R and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R (14-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 12h volume regime: current 12h volume > 2.0x 20-period MA
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    volume_spike = vol_12h > (2.0 * vol_ma_aligned)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Williams %R thresholds
    oversold = -80
    overbought = -20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        wr = williams_r_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(wr) or np.isnan(vol_spike):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R conditions (need previous value for crossover)
        if i > 100:
            wr_prev = williams_r_aligned[i-1]
            wr_oversold_cross = (wr_prev <= oversold) and (wr > oversold)
            wr_overbought_cross = (wr_prev >= overbought) and (wr < overbought)
        else:
            wr_oversold_cross = False
            wr_overbought_cross = False
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Williams %R crosses above -80 (oversold bounce) with volume spike
            long_entry = wr_oversold_cross and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Williams %R crosses below -20 (overbought bounce) with volume spike
            short_entry = wr_overbought_cross and vol_spike
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
    
    return signals