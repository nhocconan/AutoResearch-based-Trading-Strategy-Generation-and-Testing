#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and volume confirmation
# Uses 1w HTF for EMA50 to capture long-term trend and reduce false signals in ranging markets.
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3) - signals when aligned (trending) or entwined (ranging).
# Long: Lips > Teeth > Jaw AND price > 1w EMA50 AND volume > 1.5x 20-day average.
# Short: Lips < Teeth < Jaw AND price < 1w EMA50 AND volume > 1.5x 20-day average.
# Exit: When Alligator lines become entwined (not aligned) or price crosses 1w EMA50 in opposite direction.
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and fee drag.

name = "1d_WilliamsAlligator_JawTeethLips_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
    # Jaw: 13-period SMA, smoothed by 8 periods
    close_series = pd.Series(close)
    jaw_raw = close_series.rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth_raw = close_series.rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips_raw = close_series.rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean().values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 1.5x 20-period average (balanced threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips > Teeth > Jaw (aligned bullish) AND price > 1w EMA50 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (aligned bearish) AND price < 1w EMA50 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator becomes entwined (not aligned) OR price < 1w EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator becomes entwined (not aligned) OR price > 1w EMA50
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals