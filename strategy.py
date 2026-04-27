#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for daily ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-day ATR for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-day EMA for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 4h data for trend confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema10_4h = pd.Series(close_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align HTF data to 1h timeframe
    atr10_1h = align_htf_to_ltf(prices, df_1d, atr10)
    ema20_1h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema10_4h_1h = align_htf_to_ltf(prices, df_4h, ema10_4h)
    
    # Calculate 1h volatility (ATR-like)
    tr1h = high[1:] - low[1:]
    tr2h = np.abs(high[1:] - close[:-1])
    tr3h = np.abs(low[1:] - close[:-1])
    tr_h = np.concatenate([[np.nan], np.maximum(tr1h, np.maximum(tr2h, tr3h))])
    atr1h = pd.Series(tr_h).rolling(window=10, min_periods=10).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr10_1h[i]) or np.isnan(ema20_1h[i]) or np.isnan(ema10_4h_1h[i]) or 
            np.isnan(atr1h[i]) or np.isnan(close[i-1]) if i > 0 else False):
            signals[i] = 0.0
            continue
        
        # Skip outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: require sufficient volatility
        vol_filter = atr1h[i] > (atr10_1h[i] * 0.5)
        
        # Trend alignment: 1h EMA above/below 4h EMA
        uptrend_4h = close[i] > ema10_4h_1h[i]
        downtrend_4h = close[i] < ema10_4h_1h[i]
        
        # Daily trend filter
        daily_uptrend = close[i] > ema20_1h[i]
        daily_downtrend = close[i] < ema20_1h[i]
        
        # Long conditions:
        # 1. Pullback to daily EMA in uptrend with volume confirmation
        pullback_long = (daily_uptrend and 
                        close[i] <= ema20_1h[i] * 1.01 and  # Near daily EMA
                        close[i] > ema20_1h[i] * 0.99 and
                        uptrend_4h and
                        vol_filter and
                        volume[i] > np.nanmedian(volume[max(0, i-20):i]) * 1.5)
        
        # 2. Break above recent high with momentum
        recent_high = np.nanmax(high[max(0, i-12):i]) if i >= 12 else np.nan
        breakout_long = (not np.isnan(recent_high) and
                        close[i] > recent_high and
                        close[i-1] <= recent_high and
                        daily_uptrend and
                        uptrend_4h and
                        vol_filter)
        
        # Short conditions:
        # 1. Pullback to daily EMA in downtrend
        pullback_short = (daily_downtrend and 
                         close[i] >= ema20_1h[i] * 0.99 and  # Near daily EMA
                         close[i] <= ema20_1h[i] * 1.01 and
                         downtrend_4h and
                         vol_filter and
                         volume[i] > np.nanmedian(volume[max(0, i-20):i]) * 1.5)
        
        # 2. Break below recent low with momentum
        recent_low = np.nanmin(low[max(0, i-12):i]) if i >= 12 else np.nan
        breakout_short = (not np.isnan(recent_low) and
                         close[i] < recent_low and
                         close[i-1] >= recent_low and
                         daily_downtrend and
                         downtrend_4h and
                         vol_filter)
        
        if pullback_long or breakout_long:
            signals[i] = 0.20
            position = 1
        elif pullback_short or breakout_short:
            signals[i] = -0.20
            position = -1
        # Exit conditions: reverse signal or loss of trend
        elif position == 1 and (not daily_uptrend or not uptrend_4h):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not daily_downtrend or not downtrend_4h):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_DailyEMA_Pullback_Breakout_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0