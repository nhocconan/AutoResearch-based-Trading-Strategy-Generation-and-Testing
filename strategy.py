#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume confirmation.
# Long when Williams %R(14) crosses above -80 from oversold AND price > 1d EMA34 (uptrend) AND volume > 1.3x 20-bar average.
# Short when Williams %R(14) crosses below -20 from overbought AND price < 1d EMA34 (downtrend) AND volume > 1.3x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture mean reversions in ranging markets.
# Williams %R identifies overextended moves likely to reverse. 1d EMA34 ensures alignment with higher timeframe trend.
# Volume confirmation reduces false signals from low-participation reversals.
# Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.

name = "6h_WilliamsR_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]
    williams_r_cross_up = (williams_r_prev <= -80) & (williams_r > -80)
    williams_r_cross_down = (williams_r_prev >= -20) & (williams_r < -20)
    
    # Volume confirmation: current 6h volume > 1.3x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA and Williams %R calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.3)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold reversal) AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (williams_r_cross_up[i] and 
                curr_close > ema_34_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (williams_r_cross_down[i] and 
                  curr_close < ema_34_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR price < 1d EMA34 (trend change)
            if (williams_r_cross_down[i] or 
                curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR price > 1d EMA34 (trend change)
            if (williams_r_cross_up[i] or 
                curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals