#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) + 1d EMA34 trend filter + volume confirmation.
# In bull regime (price > 1d EMA34), go long when Williams %R crosses above -80 (oversold bounce).
# In bear regime (price < 1d EMA34), go short when Williams %R crosses below -20 (overbought rejection).
# Uses Williams %R for mean reversion entries within the trend, 1d EMA34 for regime filter,
# and 6h volume spike (>2x 20-period MA) for confirmation. Designed for 50-150 total trades over 4 years.
# Williams %R is underutilized and provides a clear BTC/ETH edge by capturing swing points in trends.

name = "6h_WilliamsR_1dEMA34_VolumeSpike_Trend"
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
    
    # Get 1d data for Williams %R calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h (wait for 1d bar to complete)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1d data for EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        wr_val = williams_r_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R signals: -80 oversold, -20 overbought
        wr_oversold = wr_val < -80
        wr_overbought = wr_val > -20
        wr_cross_above_oversold = (wr_val > -80) and (i > 100 and williams_r_aligned[i-1] <= -80)
        wr_cross_below_overbought = (wr_val < -20) and (i > 100 and williams_r_aligned[i-1] >= -20)
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Williams %R crosses above -80 (oversold bounce) with volume spike
            long_entry = wr_cross_above_oversold and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Williams %R crosses below -20 (overbought rejection) with volume spike
            short_entry = wr_cross_below_overbought and vol_spike
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
            if wr_val < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R crossing above -50 (momentum loss) or regime change to bull
            if wr_val > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals