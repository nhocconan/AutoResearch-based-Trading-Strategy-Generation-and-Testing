#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h trend filter and daily volume confirmation.
# Uses 12h EMA for trend direction and daily volume spike for entry timing.
# Long when price > 12h EMA and daily volume > 2x average, short when opposite.
# Includes ATR-based stop loss to limit drawdown. Designed for 20-50 trades/year.

name = "4h_12h_ema_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and daily data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA (21-period) for trend direction
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 2.0 * daily average volume
        vol_filter = volume[i] > 2.0 * vol_avg_aligned[i]
        
        # Determine trend direction
        is_uptrend = close[i] > ema_12h_aligned[i]
        is_downtrend = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        enter_long = is_uptrend and vol_filter
        enter_short = is_downtrend and vol_filter
        
        # Exit conditions (ATR-based stop loss)
        exit_long = False
        exit_short = False
        if position == 1:
            # Track entry price for stop loss (simplified: use previous close)
            if i > 0:
                exit_long = low[i] < close[i-1] - 2.5 * atr_aligned[i]
        elif position == -1:
            if i > 0:
                exit_short = high[i] > close[i-1] + 2.5 * atr_aligned[i]
        
        # Update position and signals
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals