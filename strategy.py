#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d trend filter and volume confirmation.
# Uses Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) 
# to measure bull/bear strength. Trades in direction of 1d EMA50 trend.
# Volume filter requires current volume > 1.3x 20-period average.
# Works in bull markets via bull power signals and in bear markets via bear power signals.
# Target: 100-200 trades over 4 years (25-50/year).

name = "6h_elder_ray_trend_filter_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: 13-period EMA
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if 1d trend data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma20[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power becomes negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (bear_power[i] < 0 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power becomes negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (bull_power[i] < 0 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1d trend filter
            if volume_filter:
                # Long: bull power positive and price above 1d EMA50
                if (bull_power[i] > 0 and 
                    close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bear power positive and price below 1d EMA50
                elif (bear_power[i] > 0 and 
                      close[i] < ema50_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals