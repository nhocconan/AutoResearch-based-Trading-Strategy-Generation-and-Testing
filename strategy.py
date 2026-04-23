#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversal with 1d EMA34 Trend Filter and Volume Spike
- Uses Williams %R(14) to identify oversold/overbought conditions for mean reversion entries
- 1d EMA34 defines higher timeframe trend: only long above EMA34, only short below EMA34
- Volume confirmation (> 1.8x 20-period average) ensures institutional participation
- ATR-based trailing stop: exit when price moves 2.5*ATR against position from highest close
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in bull markets via trend-aligned mean reversion and in bear markets via faded extremes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_close_since_entry = 0.0  # for trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_close_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
                highest_close_since_entry = close[i]
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
                highest_close_since_entry = close[i]
        else:
            # Update highest close since entry for trailing stop
            if position == 1:
                highest_close_since_entry = max(highest_close_since_entry, close[i])
                # Exit long: price drops 2.5*ATR below highest close OR Williams %R > -50 (exit extreme)
                if (close[i] < highest_close_since_entry - 2.5 * atr[i] or 
                    williams_r[i] > -50):
                    signals[i] = 0.0
                    position = 0
                    highest_close_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                highest_close_since_entry = min(highest_close_since_entry, close[i])
                # Exit short: price rises 2.5*ATR above lowest close OR Williams %R < -50 (exit extreme)
                if (close[i] > highest_close_since_entry + 2.5 * atr[i] or 
                    williams_r[i] < -50):
                    signals[i] = 0.0
                    position = 0
                    highest_close_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_ATR_Trail"
timeframe = "4h"
leverage = 1.0