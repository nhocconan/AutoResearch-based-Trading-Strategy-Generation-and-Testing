#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 1w EMA34), go long when Williams %R crosses above -80 from below with volume spike.
# In bear regime (price < 1w EMA34), go short when Williams %R crosses below -20 from above with volume spike.
# Uses Williams %R from 1d for mean reversion entries, 1w EMA34 for regime filter, and 1d volume spike for confirmation.
# Designed for 30-100 total trades over 4 years. Focus on BTC/ETH as primary symbols.

name = "1d_WilliamsR_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = williams_r[0] if not np.isnan(williams_r[0]) else -50
    
    for i in range(14, n):
        # Get current values
        close_val = close[i]
        williams_r_val = williams_r[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(williams_r_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_williams_r = williams_r_val
            continue
            
        # Determine regime: bull if close > 1w EMA34, bear if close < 1w EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R crossover conditions
        williams_r_cross_up = (prev_williams_r < -80) and (williams_r_val >= -80)
        williams_r_cross_down = (prev_williams_r > -20) and (williams_r_val <= -20)
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Williams %R crosses above -80 from below with volume spike
            long_entry = williams_r_cross_up and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Williams %R crosses below -20 from above with volume spike
            short_entry = williams_r_cross_down and vol_spike
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
            # Exit on Williams %R crossing above -20 (overbought) or regime change to bear
            if williams_r_val > -20 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R crossing below -80 (oversold) or regime change to bull
            if williams_r_val < -80 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        prev_williams_r = williams_r_val
    
    return signals