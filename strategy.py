#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h Bollinger Bands and 1d trend filter.
# Uses 4h Bollinger Bands for mean reversion signals: long when price touches lower band in uptrend,
# short when price touches upper band in downtrend. 1d EMA50 determines trend direction.
# Includes volume confirmation (1.5x average volume) and session filter (08-20 UTC).
# Designed for low trade frequency (<40/year) to minimize fee drag in choppy markets.
name = "1h_BollingerMeanReversion_4hBB_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Bollinger Bands on 4h close
    close_4h = df_4h['close'].values
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma_20_4h + 2.0 * std_20_4h
    lower_bb_4h = sma_20_4h - 2.0 * std_20_4h
    
    # Align Bollinger Bands to 1h timeframe
    upper_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_bb_4h)
    lower_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_bb_4h)
    sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_20_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for EMA50 and 20 for Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_bb_4h_aligned[i]) or np.isnan(lower_bb_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(sma_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_bb = upper_bb_4h_aligned[i]
        lower_bb = lower_bb_4h_aligned[i]
        sma_20 = sma_20_4h_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Price touches lower BB AND price > 1d EMA50 (uptrend) AND volume > 1.5x average
            if close[i] <= lower_bb and close[i] > ema_50 and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Enter short: Price touches upper BB AND price < 1d EMA50 (downtrend) AND volume > 1.5x average
            elif close[i] >= upper_bb and close[i] < ema_50 and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses above SMA(20) OR trend reverses (price < 1d EMA50)
            if close[i] >= sma_20 or close[i] < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses below SMA(20) OR trend reverses (price > 1d EMA50)
            if close[i] <= sma_20 or close[i] > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals