#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily trend following with weekly trend filter and volume confirmation.
# Long when price > daily EMA(34), weekly EMA(34) confirms uptrend, and volume > 1.5x daily 20-period average.
# Short when price < daily EMA(34), weekly EMA(34) confirms downtrend, and volume > 1.5x daily 20-period average.
# Exit when price crosses back below/above daily EMA(34).
# Designed for ~10-30 trades/year per symbol with low turnover.
name = "1d_EMA34_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily EMA(34) for entry trigger
    ema_34_daily = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 1.5 * daily 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_34_daily[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        daily_ema_val = ema_34_daily[i]
        weekly_ema_val = ema_34_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price above daily EMA, weekly EMA confirms uptrend, volume surge
            if close_val > daily_ema_val and weekly_ema_val > daily_ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA, weekly EMA confirms downtrend, volume surge
            elif close_val < daily_ema_val and weekly_ema_val < daily_ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below daily EMA
            if close_val < daily_ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above daily EMA
            if close_val > daily_ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals