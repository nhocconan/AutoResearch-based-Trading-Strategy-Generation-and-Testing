#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with weekly trend filter and volume confirmation.
# Elder Ray calculates Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# Long when Bull Power > 0 and Bear Power rising (less negative), short when Bear Power < 0 and Bull Power falling.
# Weekly trend filter (price above/below 20-week EMA) ensures alignment with higher timeframe trend.
# Volume confirmation (current volume > 1.3x 20-period average) filters low-quality signals.
# Works in bull markets via bull power strength and in bear markets via bear power exhaustion.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_elder_ray_trend_filter_v4"
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
    
    # Elder Ray: EMA13 for Bull/Bear Power
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: 20-week EMA on weekly closes
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20w = np.full(len(close_1w), np.nan)
    for i in range(19, len(close_1w)):
        if i == 19:
            ema_20w[i] = np.mean(close_1w[0:20])
        else:
            ema_20w[i] = close_1w[i] * 2/(20+1) + ema_20w[i-1] * (1 - 2/(20+1))
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if weekly trend data not available
        if np.isnan(ema_20w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power becomes positive (weakening uptrend) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (bear_power[i] > 0 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power becomes negative (weakening downtrend) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (bull_power[i] < 0 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Long: bull power positive and rising, bear power negative but improving (less negative)
                if (bull_power[i] > 0 and 
                    bear_power[i] < 0 and 
                    bear_power[i] > bear_power[i-1] and  # bear power rising (less negative)
                    close[i] > ema_20w_aligned[i]):  # weekly uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bear power negative and falling, bull power positive but deteriorating (less positive)
                elif (bear_power[i] < 0 and 
                      bull_power[i] > 0 and 
                      bull_power[i] < bull_power[i-1] and  # bull power falling (less positive)
                      close[i] < ema_20w_aligned[i]):  # weekly downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals