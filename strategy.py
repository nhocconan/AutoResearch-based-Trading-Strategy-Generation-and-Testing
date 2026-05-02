#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Works in bull markets via pullbacks to EMA in uptrends, in bear via bounces from oversold in downtrends
# Williams %R identifies extreme conditions; EMA34 filters trend direction; volume spike confirms participation
# Designed with tight entry conditions to target 50-150 total trades over 4 years (12-37/year)
# Uses proven structure: momentum extreme + trend filter + volume confirmation

name = "12h_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R(14) from 12h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) + price > 1d EMA34 + volume spike
            if williams_r[i] < -80 and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 1d EMA34 + volume spike
            elif williams_r[i] > -20 and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or price < 1d EMA34
            if williams_r[i] > -20 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or price > 1d EMA34
            if williams_r[i] < -80 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals