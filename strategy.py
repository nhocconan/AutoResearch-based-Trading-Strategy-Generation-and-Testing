#!/usr/bin/env python3
"""
Hypothesis: 1-hour range trading with 4-hour trend filter and volume confirmation.
Trades mean reversion at 1-hour Bollinger Bands (20,2) when 4-hour trend is strong.
In bull markets: buy dips in uptrend. In bear markets: sell rallies in downtrend.
Uses volume filter to avoid low-liquidity whipsaws.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
Position size: 0.20 (20%) to manage drawdown.
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
    
    # Get 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(34) for trend
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 4-hour data for volume filter
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate 1-hour Bollinger Bands (20,2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_middle_vals = bb_middle.values
    bb_upper_vals = bb_upper.values
    bb_lower_vals = bb_lower.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need 4h EMA, volume MA, and BB
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or
            np.isnan(bb_middle_vals[i]) or np.isnan(bb_upper_vals[i]) or np.isnan(bb_lower_vals[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_34_4h_val = ema_34_4h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        bb_upper = bb_upper_vals[i]
        bb_middle = bb_middle_vals[i]
        bb_lower = bb_lower_vals[i]
        
        # Volume filter: volume > 1.3x 4h average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Trend determination from 4h EMA
        uptrend = close[i] > ema_34_4h_val
        downtrend = close[i] < ema_34_4h_val
        
        # Entry conditions: mean reversion with trend filter
        if position == 0:
            # Long: pullback to lower BB in uptrend with volume
            if close[i] <= bb_lower and uptrend and vol_filter:
                signals[i] = size
                position = 1
            # Short: rally to upper BB in downtrend with volume
            elif close[i] >= bb_upper and downtrend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to middle BB or trend reversal
            if close[i] >= bb_middle or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: return to middle BB or trend reversal
            if close[i] <= bb_middle or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_BollingerMeanReversion_4hTrendFilter_Volume"
timeframe = "1h"
leverage = 1.0