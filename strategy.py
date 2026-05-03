#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from below (oversold bounce), price > 1d EMA34, and volume > 1.5x 20-bar average
# Short when Williams %R crosses below -20 from above (overbought rejection), price < 1d EMA34, and volume > 1.5x 20-bar average
# Uses Williams %R for mean reversion timing, 1d EMA34 for trend filter, volume for momentum confirmation
# Designed for low-moderate trade frequency (~19-50/year on 4h) to minimize fee drag
# Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend)

name = "4h_WilliamsR_Volume_1dEMA34_Trend_Volume_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R (14-period) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation (1.5x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    williams_r_cross_up = (williams_r_prev <= -80) & (williams_r > -80)  # crossed above -80
    williams_r_cross_down = (williams_r_prev >= -20) & (williams_r < -20)  # crossed below -20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 14, 20) + 1  # EMA34(1d) + Williams %R(14) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(williams_r_cross_up[i]) or 
            np.isnan(williams_r_cross_down[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crossed above -80, close > 1d EMA34, volume spike
            if (williams_r_cross_up[i] and 
                close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crossed below -20, close < 1d EMA34, volume spike
            elif (williams_r_cross_down[i] and 
                  close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) or close < 1d EMA34 (trend failure)
            if (williams_r_cross_down[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) or close > 1d EMA34 (trend failure)
            if (williams_r_cross_up[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals