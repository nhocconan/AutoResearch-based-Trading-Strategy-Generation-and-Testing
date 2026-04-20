#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 4h Williams %R overbought/oversold + 1d trend filter (EMA50) + volume confirmation.
# Long when Williams %R crosses above -50 from below, price > 1d EMA50, volume > 1.5x 20-period average.
# Short when Williams %R crosses below -50 from above, price < 1d EMA50, volume > 1.5x 20-period average.
# Williams %R identifies short-term reversals; 1d EMA50 filters counter-trend trades.
# Volume confirmation ensures institutional participation.
# Target: 25-35 trades/year per symbol (100-140 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 4h data for Williams %R
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        wr = williams_r[i]
        vol_ok = vol_filter[i]
        
        # Williams %R cross above -50 (bullish momentum)
        wr_cross_up = wr > -50 and (i == 100 or williams_r[i-1] <= -50)
        # Williams %R cross below -50 (bearish momentum)
        wr_cross_down = wr < -50 and (i == 100 or williams_r[i-1] >= -50)
        
        if position == 0:
            # Long: Williams %R crosses above -50, price above 1d EMA50, volume
            if wr_cross_up and price > ema_50 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50, price below 1d EMA50, volume
            elif wr_cross_down and price < ema_50 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -50 or price below 1d EMA50
            if wr_cross_down or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -50 or price above 1d EMA50
            if wr_cross_up or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1d_EMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0