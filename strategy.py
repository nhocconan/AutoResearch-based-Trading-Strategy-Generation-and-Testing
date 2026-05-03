#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 1d EMA34), go long when Williams %R crosses above -80 (oversold recovery) with volume spike.
# In bear regime (price < 1d EMA34), go short when Williams %R crosses below -20 (overbought breakdown) with volume spike.
# Uses Williams %R(14) for momentum, 1d EMA34 for regime filter, and 4h volume spike for confirmation.
# Designed for 75-200 total trades over 4 years (19-50/year) on BTC/ETH.

name = "4h_WilliamsR_1dEMA34_VolumeSpike_Trend"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R(14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
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
        
        # Williams %R signals with crossover detection
        wr_long_signal = (wr > -80) and (williams_r[i-1] <= -80) if i > 0 else False
        wr_short_signal = (wr < -20) and (williams_r[i-1] >= -20) if i > 0 else False
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Williams %R crosses above -80 (oversold recovery) with volume spike
            long_entry = wr_long_signal and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Williams %R crosses below -20 (overbought breakdown) with volume spike
            short_entry = wr_short_signal and vol_spike
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
            # Exit on Williams %R crosses below -50 (momentum loss) or regime change to bear
            wr_exit_signal = (wr < -50) and (williams_r[i-1] >= -50) if i > 0 else False
            if wr_exit_signal or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R crosses above -50 (momentum loss) or regime change to bull
            wr_exit_signal = (wr > -50) and (williams_r[i-1] <= -50) if i > 0 else False
            if wr_exit_signal or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals