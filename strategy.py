#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Williams %R > -20 = overbought, < -80 = oversold.
# Long when: Williams %R crosses above -80 from below AND price > 12h EMA(34) AND volume > 1.5x 20-period average.
# Short when: Williams %R crosses below -20 from above AND price < 12h EMA(34) AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back through -50.
# Uses mean reversion in extreme zones with trend filter to avoid counter-trend trades.
# Designed for ~15-25 trades/year per symbol (60-100 total over 4 years).
name = "6h_WilliamsR_12hEMA34_Volume"
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
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_34_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_34)
    
    # Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_12h = ema_12h_34_aligned[i]
        vol_filter = volume_filter[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below, price above 12h EMA34, high volume
            if wr_prev <= -80 and wr > -80 and price > ema_12h and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, price below 12h EMA34, high volume
            elif wr_prev >= -20 and wr < -20 and price < ema_12h and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if wr_prev <= -50 and wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if wr_prev >= -50 and wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals