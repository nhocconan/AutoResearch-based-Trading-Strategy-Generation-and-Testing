#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 1d EMA34), go long when Williams %R crosses above -80 from oversold with volume spike.
# In bear regime (price < 1d EMA34), go short when Williams %R crosses below -20 from overbought with volume spike.
# Uses 1d EMA34 for regime filter, 6h Williams %R (14-period) for mean reversion entries,
# and 6h volume spike for confirmation. Designed for 50-150 total trades over 4 years.
# Focus on BTC/ETH as primary symbols.

name = "6h_WilliamsR_1dEMA34_VolumeSpike_MR"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        wr = williams_r[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R cross conditions
        wr_cross_up = (wr > -80) and (i > 100 and williams_r[i-1] <= -80)
        wr_cross_down = (wr < -20) and (i > 100 and williams_r[i-1] >= -20)
        
        # Regime-based entry conditions
        if position == 0:
            if is_bull_regime and wr_cross_up and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_regime and wr_cross_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on Williams %R cross below -50 (momentum loss) or regime change to bear
            if (wr < -50 and i > 100 and williams_r[i-1] >= -50) or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R cross above -50 (momentum loss) or regime change to bull
            if (wr > -50 and i > 100 and williams_r[i-1] <= -50) or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals