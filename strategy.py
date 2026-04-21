#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) + 12h EMA(50) Trend Filter + 1d Volume Spike
# Long when Williams %R crosses above -20 from below and price > 12h EMA(50) and 1d volume > 1.5x 20-period average
# Short when Williams %R crosses below -80 from above and price < 12h EMA(50) and 1d volume > 1.5x 20-period average
# Exit when Williams %R crosses back below -50 (long) or above -50 (short)
# Williams %R identifies overbought/oversold conditions; EMA filter ensures trend alignment; volume confirms momentum
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d volume moving average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams %R (14) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when HH == LL)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = prices['close'].iloc[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        # Volume confirmation: 1d volume > 1.5x 20-period average
        vol_ma = vol_ma_20_1d_aligned[i]
        vol_1d_current = df_1d['volume'].iloc[i // 2] if i >= 2 else df_1d['volume'].iloc[0]  # 2x 6h bars per 12h, so 12 per 1d
        volume_confirm = vol_1d_current > 1.5 * vol_ma
        
        # Trend filter: price relative to 12h EMA(50)
        price_above_ema = price > ema_50_12h_aligned[i]
        price_below_ema = price < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long entry: WR crosses above -20 from below, price above EMA, volume confirmation
            if wr > -20 and wr_prev <= -20 and price_above_ema and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: WR crosses below -80 from above, price below EMA, volume confirmation
            elif wr < -80 and wr_prev >= -80 and price_below_ema and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # long position
            # Exit when WR crosses back below -50
            if wr < -50 and wr_prev >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # hold long
        
        elif position == -1:  # short position
            # Exit when WR crosses back above -50
            if wr > -50 and wr_prev <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # hold short
    
    return signals

name = "6h_WilliamsR14_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0