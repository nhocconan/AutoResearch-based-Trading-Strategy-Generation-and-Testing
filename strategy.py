#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold), close > 1d EMA34, volume > 1.8x 20-bar average
# Short when Williams %R > -20 (overbought), close < 1d EMA34, volume > 1.8x 20-bar average
# Williams %R identifies exhaustion points, 1d EMA34 ensures trend alignment, volume confirms momentum
# Designed for low trade frequency (~12-37/year on 6h) to minimize fee drag
# Works in bull (mean reversion during uptrend pullbacks) and bear (mean reversion during downtrend bounces)

name = "6h_WilliamsR_Volume_1dEMA34_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d timeframe (period=14)
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_1d - df_1d['close'].values) / (highest_high_1d - lowest_low_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Volume confirmation (1.8x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20) + 1  # EMA34(1d) + Williams %R(14) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold), close > 1d EMA34, volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought), close < 1d EMA34, volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or close < 1d EMA34 (trend failure)
            if (williams_r_aligned[i] > -20 or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or close > 1d EMA34 (trend failure)
            if (williams_r_aligned[i] < -80 or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals