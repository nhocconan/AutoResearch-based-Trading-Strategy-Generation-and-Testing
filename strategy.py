#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Uses 13-period EMA as the trend reference.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling (bullish momentum).
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising (bearish momentum).
# 1d trend filter: price above/below 50-period EMA on daily chart.
# Volume confirmation: current volume > 1.3x 20-period average.
# Works in bull markets via bullish momentum and in bear markets via bearish momentum.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_elder_ray_trend_filter_v2"
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
    
    # Elder Ray components: EMA(13) of close
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema13
    # Bear Power = EMA(13) - Low
    bear_power = ema13 - low
    
    # Slope of Bull Power and Bear Power (1-period change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    
    # 1d trend filter: 50-period EMA on daily chart
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(13, n):
        # Skip if 1d trend data not available
        if np.isnan(ema_50d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bull Power turns negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (bull_power[i] <= 0 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power turns negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (bear_power[i] <= 0 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1d trend filter
            if volume_filter:
                # Long: Bull Power > 0 and rising, price above 1d EMA50
                if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                    close[i] > ema_50d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bear Power > 0 and rising, price below 1d EMA50
                elif (bear_power[i] > 0 and bear_power_slope[i] > 0 and 
                      close[i] < ema_50d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals