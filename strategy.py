#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversal + 1w Supertrend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions on daily timeframe;
# weekly Supertrend filters for higher timeframe trend direction to avoid counter-trend trades;
# volume confirmation ensures breakout/mean reversion has participation;
# Target: 20-30 trades/year (80-120 total over 4 years) to balance opportunity and fee drag.

name = "1d_WilliamsR_MeanRev_1wSupertrend_VolumeConfirm_v1"
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
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Supertrend for trend filter
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['close']).shift()).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close']).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_vals = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    basic_ub = (df_1w['high'] + df_1w['low']) / 2 + multiplier * atr_vals
    basic_lb = (df_1w['high'] + df_1w['low']) / 2 - multiplier * atr_vals
    
    # Final Upper and Lower Bands
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(df_1w)):
        if basic_ub.iloc[i] < final_ub.iloc[i-1] or df_1w['close'].iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]
            
        if basic_lb.iloc[i] > final_lb.iloc[i-1] or df_1w['close'].iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]
    
    # Supertrend direction: 1 for uptrend, -1 for downtrend
    supertrend_dir = np.ones(len(df_1w), dtype=float) * np.nan
    for i in range(len(df_1w)):
        if i == 0:
            supertrend_dir[i] = 1.0  # start with uptrend assumption
        else:
            if supertrend_dir[i-1] == 1.0 and df_1w['close'].iloc[i] <= final_ub.iloc[i]:
                supertrend_dir[i] = -1.0
            elif supertrend_dir[i-1] == -1.0 and df_1w['close'].iloc[i] >= final_lb.iloc[i]:
                supertrend_dir[i] = 1.0
            else:
                supertrend_dir[i] = supertrend_dir[i-1]
    
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir)
    
    # Calculate Williams %R(14) for mean reversion signals
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(14, 20)  # warmup for Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(supertrend_dir_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_williams_r = williams_r[i]
        curr_supertrend_dir = supertrend_dir_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Williams %R rises above -20 (overbought)
            # 2. Weekly Supertrend turns down (trend change)
            if (curr_williams_r > -20 or curr_supertrend_dir == -1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Williams %R falls below -80 (oversold)
            # 2. Weekly Supertrend turns up (trend change)
            if (curr_williams_r < -80 or curr_supertrend_dir == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R below -80 (oversold) + weekly uptrend + volume confirm
            if (curr_williams_r < -80 and
                curr_supertrend_dir == 1.0 and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Williams %R above -20 (overbought) + weekly downtrend + volume confirm
            elif (curr_williams_r > -20 and
                  curr_supertrend_dir == -1.0 and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals