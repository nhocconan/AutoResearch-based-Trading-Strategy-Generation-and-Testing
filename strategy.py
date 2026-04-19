#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h trend filter and volume confirmation.
# Long when: Williams %R < -80 (oversold) and price > 12h EMA(34) and volume > 1.5x average
# Short when: Williams %R > -20 (overbought) and price < 12h EMA(34) and volume > 1.5x average
# Exit when Williams %R crosses back to -50.
# Williams %R identifies reversals in both bull and bear markets. EMA filter ensures trend alignment.
# Volume confirmation reduces false signals. Designed for ~20-30 trades/year per symbol.
name = "4h_WilliamsR_EMA34_Volume"
timeframe = "4h"
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
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_12h = ema_12h_34_aligned[i]
        wr = williams_r[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80), price above 12h EMA34, volume confirmation
            if wr < -80 and price > ema_12h and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below 12h EMA34, volume confirmation
            elif wr > -20 and price < ema_12h and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals