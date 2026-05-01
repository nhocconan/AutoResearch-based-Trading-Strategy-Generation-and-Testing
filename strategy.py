#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) mean reversion with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume > 1.3x 20-bar average.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume > 1.3x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture swing reversals.
# Williams %R identifies overextended moves ripe for mean reversion.
# 1d EMA34 ensures trades align with higher timeframe momentum.
# Volume confirmation filters low-momentum false signals.
# Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.

name = "12h_WilliamsR_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Calculate Williams %R (14) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Volume confirmation: current 12h volume > 1.3x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA and Williams %R calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        curr_williams_r = williams_r[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.3)
        
        # Williams %R signals
        oversold = curr_williams_r < -80
        overbought = curr_williams_r > -20
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (oversold and 
                curr_close > ema_34_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (overbought and 
                  curr_close < ema_34_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below 1d EMA34 (trend change) OR Williams %R > -50 (exit oversold)
            if (curr_close < ema_34_aligned[i] or 
                curr_williams_r > -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA34 (trend change) OR Williams %R < -50 (exit overbought)
            if (curr_close > ema_34_aligned[i] or 
                curr_williams_r < -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals