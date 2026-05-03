#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold reversal) with volume > 1.5x 20-bar average and close > 1d EMA34 (uptrend)
# Short when Williams %R crosses below -20 (overbought reversal) with volume > 1.5x 20-bar average and close < 1d EMA34 (downtrend)
# Exit when Williams %R crosses above -20 (long exit) or below -80 (short exit) or trend fails
# Williams %R identifies reversal points in ranging markets; works in both bull (buy dips) and bear (sell rallies)
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.30) to minimize fee churn.

name = "4h_WilliamsR_1dEMA34_Volume_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 14, 20) + 1  # EMA34(1d) + Williams %R(14) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (from below) with volume spike and close > 1d EMA34 (uptrend)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: Williams %R crosses below -20 (from above) with volume spike and close < 1d EMA34 (downtrend)
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) or close < 1d EMA34 (trend failure)
            if (williams_r[i] >= -20 and williams_r[i-1] < -20) or \
               close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) or close > 1d EMA34 (trend failure)
            if (williams_r[i] <= -80 and williams_r[i-1] > -80) or \
               close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals