#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR-based volatility filter and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align volatility and trend filters to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h ATR(10) for stoploss calculation
    tr_4h = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr_4h = np.maximum(np.abs(low[1:] - close[:-1]), tr_4h)
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_10_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # 4-period average true range for volatility breakout
    atr_avg_4 = pd.Series(tr_4h).rolling(window=4, min_periods=4).mean().values
    
    # 20-period volume average for volume confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_10_4h[i]) or np.isnan(atr_avg_4[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility condition: 4h ATR(4) > 1.5 * daily ATR(14) (volatility expansion)
        vol_expansion = atr_avg_4[i] > 1.5 * atr_14_aligned[i]
        
        if position == 0:
            # Long: Close above EMA50 with volatility expansion and volume spike
            if (close[i] > ema_50_aligned[i] and vol_expansion and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below EMA50 with volatility expansion and volume spike
            elif (close[i] < ema_50_aligned[i] and vol_expansion and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Stoploss: 2 * ATR(10) from entry price (tracked via position direction)
            if position == 1:
                # Long stoploss: close below entry - 2*ATR
                # We approximate entry as the close when position was opened
                # For simplicity, use a trailing stop based on recent high
                recent_high = np.maximum.accumulate(high[:i+1])[-1]
                if close[i] < recent_high - 2.0 * atr_10_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short stoploss: close above entry + 2*ATR
                recent_low = np.minimum.accumulate(low[:i+1])[-1]
                if close[i] > recent_low + 2.0 * atr_10_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_VolatilityBreakout_EMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0