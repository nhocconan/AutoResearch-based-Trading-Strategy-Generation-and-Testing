#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-compute for speed)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for trend filter
    close_4h = df_4h['close'].values
    delta = pd.Series(close_4h).diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_vals = rsi_4h.values
    
    # Calculate 4h ATR(14) for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])))
    tr3 = pd.Series(np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_4h = tr_4h.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe with proper delay
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_vals)
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 1h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h RSI > 50 (bullish trend) + 1h price breaks above 20-period high + volume confirmation → long
        # 2. 4h RSI < 50 (bearish trend) + 1h price breaks below 20-period low + volume confirmation → short
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.20
        
        # Long conditions: 4h bullish trend + 1h breakout above high
        if (rsi_4h_aligned[i] > 50 and            # 4h bullish trend
            close[i] > highest_20[i] and          # 1h price above 20-period high
            volume_ratio[i] > 1.5 and             # Volume confirmation
            atr_14_4h_aligned[i] > 0.003 * close[i]):  # Volatility filter
            signals[i] = 0.20
            
        # Short conditions: 4h bearish trend + 1h breakdown below low
        elif (rsi_4h_aligned[i] < 50 and          # 4h bearish trend
              close[i] < lowest_20[i] and         # 1h price below 20-period low
              volume_ratio[i] > 1.5 and           # Volume confirmation
              atr_14_4h_aligned[i] > 0.003 * close[i]):  # Volatility filter
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_RSI_Trend_Donchian_Breakout_Volume_Filter_Session"
timeframe = "1h"
leverage = 1.0