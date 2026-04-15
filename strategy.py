#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h EMA trend filter and volume confirmation
# Uses 4h EMA(20) for trend direction (bull/bear) and 1h EMA(8/21) for entry timing.
# Volume > 1.5x 20-period median ensures institutional participation.
# Trades only during 08-20 UTC to avoid low-liquidity Asian session.
# Conservative position size (0.20) to manage drawdown in volatile markets.
# Designed to work in both bull and bear markets by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA(20) for trend direction - calculated ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h EMA(8) and EMA(21) for entry timing
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(21, n):  # Start after warmup for EMA(21)
        # Skip if outside trading session
        if not in_session[i]:
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_8[i]) or 
            np.isnan(ema_21[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Determine trend from 4h EMA
        uptrend = ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1]
        downtrend = ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1]
        
        # Long: uptrend on 4h + EMA(8) crosses above EMA(21) on 1h + volume
        if (uptrend and ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1] and
            volume[i] > vol_threshold[i]):
            signals[i] = 0.20
        
        # Short: downtrend on 4h + EMA(8) crosses below EMA(21) on 1h + volume
        elif (downtrend and ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1] and
              volume[i] > vol_threshold[i]):
            signals[i] = -0.20
        
        # Exit: EMA crossover in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and ema_8[i] < ema_21[i]) or
               (signals[i-1] == -0.20 and ema_8[i] > ema_21[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_EMACrossover_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0