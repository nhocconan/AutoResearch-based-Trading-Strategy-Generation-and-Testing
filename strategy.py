#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions. In bull regime (price > 1d EMA34),
# we go long when Williams %R crosses above -80 from below (oversold bounce). 
# In bear regime (price < 1d EMA34), we go short when Williams %R crosses below -20 from above (overbought rejection).
# Volume spike confirms momentum behind the move. This combines mean reversion entries with trend filtering
# to work in both bull and bear markets while avoiding chop.

name = "6h_WilliamsR_1dTrend_VolumeSpike_Regime"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R (14-period) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        wr_val = williams_r[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(wr_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R crossover conditions (using previous bar to avoid look-ahead)
        if i > 100:
            wr_prev = williams_r[i-1]
            # Long: WR crosses above -80 from below (oversold bounce)
            long_crossover = (wr_prev <= -80) and (wr_val > -80)
            # Short: WR crosses below -20 from above (overbought rejection)
            short_crossover = (wr_prev >= -20) and (wr_val < -20)
        else:
            long_crossover = False
            short_crossover = False
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Oversold bounce in bull trend with volume spike
            long_entry = long_crossover and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Overbought rejection in bear trend with volume spike
            short_entry = short_crossover and vol_spike
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
            # Exit on WR crossing above -20 (overbought) or regime change to bear
            if wr_val >= -20 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on WR crossing below -80 (oversold) or regime change to bull
            if wr_val <= -80 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals