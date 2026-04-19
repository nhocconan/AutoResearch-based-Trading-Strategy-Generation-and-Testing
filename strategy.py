#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (1w EMA50),
# we take counter-trend entries at extreme %R levels with volume confirmation.
# Works in bull/bear by filtering counter-trend trades during weak trends.
# Target: 25-35 trades/year per symbol.
name = "12h_WilliamsR_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R (14-period)
    def williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        
        for i in range(len(high)):
            if i < period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        wr = np.zeros_like(close)
        wr[:period-1] = np.nan
        wr[period-1:] = -100 * (highest_high[period-1:] - close[period-1:]) / (highest_high[period-1:] - lowest_low[period-1:])
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.3x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        wr_val = wr[i]
        vol_ma = vol_ma_30[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Enter long when oversold (-80 or below), above weekly EMA50, and volume confirmation
            if wr_val <= -80 and price > ema_50_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short when overbought (-20 or above), below weekly EMA50, and volume confirmation
            elif wr_val >= -20 and price < ema_50_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R returns to -50 or price crosses below weekly EMA50
            if wr_val >= -50 or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R returns to -50 or price crosses above weekly EMA50
            if wr_val <= -50 or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals