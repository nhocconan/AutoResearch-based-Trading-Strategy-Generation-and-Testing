#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA200 Trend Filter
# Uses Williams %R for overbought/oversold reversals in trending markets
# 1d EMA200 ensures we only trade in the direction of the daily trend
# Works in bull/bear by fading extremes within the trend
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200  # for EMA200 calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade long when price > daily EMA200, short when price < daily EMA200
        if price > ema_200_1d_aligned[i]:
            # Uptrend: look for oversold conditions to go long
            if position == 0:
                if williams_r[i] < -80 and vol > 1.5 * avg_vol[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long when overbought or trend changes
                if williams_r[i] > -20 or price < ema_200_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Close short if trend turns up
                position = 0
                signals[i] = 0.0
        else:
            # Downtrend: look for overbought conditions to go short
            if position == 0:
                if williams_r[i] > -20 and vol > 1.5 * avg_vol[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == -1:
                # Exit short when oversold or trend changes
                if williams_r[i] < -80 or price > ema_200_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            elif position == 1:
                # Close long if trend turns down
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_1dEMA200_Trend"
timeframe = "4h"
leverage = 1.0