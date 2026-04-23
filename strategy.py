#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND close > 1d EMA50 AND volume > 1.5x average.
Short when Williams %R crosses below -20 (overbought) AND close < 1d EMA50 AND volume > 1.5x average.
Exit when Williams %R returns to -50 (mean reversion) OR volume drops below average.
Williams %R identifies reversals in bear markets; EMA50 filters trend direction; volume confirms conviction.
Targets 12-25 trades/year per symbol (50-100 total over 4 years) to minimize fee drag.
Uses discrete position sizing (0.25) to avoid churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        ema50_val = ema50_1d_aligned[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND price > 1d EMA50 AND volume spike
            if (wr > -80 and wr_prev <= -80 and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND price < 1d EMA50 AND volume spike
            elif (wr < -20 and wr_prev >= -20 and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to -50 (mean reversion) OR volume drops below average
                if (wr >= -50 or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to -50 (mean reversion) OR volume drops below average
                if (wr <= -50 or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0