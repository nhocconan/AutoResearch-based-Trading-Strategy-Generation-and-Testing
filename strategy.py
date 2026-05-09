#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_DualTrend_VolumeBreakout"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    tr1 = high_1d - low_1d
    tr2 = (high_1d - close_1d.shift(1)).abs()
    tr3 = (low_1d - close_1d.shift(1)).abs()
    tr1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = tr1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h ATR(14) for entry filter
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr1_h[0]  # first bar
    tr2_h[0] = tr2_h[0]  # first bar
    tr3_h[0] = tr3_h[0]  # first bar
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr14_h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(atr14_1d[i]) or 
            np.isnan(atr14_h[i]) or np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        vol_ok = volume[i] > 1.5 * np.mean(volume[max(0, i-20):i+1]) if i >= 20 else False
        
        # Trend alignment: price above/below 4h EMA20
        price_above_4h_ema = close[i] > ema20_4h_aligned[i]
        price_below_4h_ema = close[i] < ema20_4h_aligned[i]
        
        # Volatility regime: only trade when 1h ATR > 0.5 * 1d ATR (enough volatility)
        vol_regime_ok = atr14_h[i] > 0.5 * atr14_1d[i]
        
        if position == 0 and in_session and vol_ok and vol_regime_ok:
            # Long: Price touches lower BB and 4h trend is up
            if close[i] <= lower_bb[i] and price_above_4h_ema:
                signals[i] = 0.20
                position = 1
            # Short: Price touches upper BB and 4h trend is down
            elif close[i] >= upper_bb[i] and price_below_4h_ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses above SMA20 or 4h trend turns down
            if close[i] >= sma20[i] or not price_above_4h_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses below SMA20 or 4h trend turns up
            if close[i] <= sma20[i] or not price_below_4h_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals