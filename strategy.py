#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) signal potential reversals.
# Long when %R crosses above -80 from below with volume > 1.5x 20-period MA and close > 1d EMA50 (uptrend).
# Short when %R crosses below -20 from above with volume spike and close < 1d EMA50 (downtrend).
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R is a momentum oscillator that works well in ranging markets; EMA50 filters counter-trend trades in trending markets.
# Volume confirmation reduces false signals. Effective in both bull and bear markets via trend alignment.

name = "6h_WilliamsR_1dEMA50_Volume"
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
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Lookback period of 14 periods
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.where(hl_range != 0, ((highest_high - close) / hl_range) * -100, -50)
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Williams %R signals: long when crossing above -80 from below, short when crossing below -20 from above
        wr_long_signal = (wr > -80) and (i == 0 or williams_r[i-1] <= -80)
        wr_short_signal = (wr < -20) and (i == 0 or williams_r[i-1] >= -20)
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 with volume spike in uptrend
            if wr_long_signal and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Williams %R crosses below -20 with volume spike in downtrend
            elif wr_short_signal and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR trend turns down
            if (wr < -50) or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR trend turns up
            if (wr > -50) or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals