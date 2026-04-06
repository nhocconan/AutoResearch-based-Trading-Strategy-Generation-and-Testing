#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d EMA trend filter + volume confirmation.
# Elder Ray measures bull/bear power relative to 13-period EMA.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# We go long when Bull Power > 0 and Bear Power < 0 (bullish momentum).
# We go short when Bear Power > 0 and Bull Power < 0 (bearish momentum).
# 1d EMA(50) filter ensures we only trade in direction of higher timeframe trend.
# Volume confirmation (current volume > 1.5x 20-period average) filters low-quality signals.
# Works in bull markets via long signals and in bear markets via short signals.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "6h_elder_ray_trend_filter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: 13-period EMA
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 49:
            ema50_1d[i] = np.nan
        elif i == 49:
            ema50_1d[i] = np.mean(close_1d[0:50])
        else:
            ema50_1d[i] = close_1d[i] * 2/(50+1) + ema50_1d[i-1] * (1 - 2/(50+1))
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power becomes positive or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (bear_power[i] > 0 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power becomes positive or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (bull_power[i] > 0 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: bull power positive, bear power negative, price above 1d EMA50
                if (bull_power[i] > 0 and bear_power[i] < 0 and 
                    close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bear power positive, bull power negative, price below 1d EMA50
                elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                      close[i] < ema50_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals