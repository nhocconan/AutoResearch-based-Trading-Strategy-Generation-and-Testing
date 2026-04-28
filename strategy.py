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
    
    # Get daily data for HTF context (ATR-based volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR(14) - Daily volatility measure
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # 4h Bollinger Bands (20, 2) for mean reversion signals
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(sma20[i]) or np.isnan(std20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR > 1.5 * 50-period average)
        atr_ma_50 = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_14_aligned[i] > 1.5 * atr_ma_50[i] if not np.isnan(atr_ma_50[i]) else False
        
        # Trend filter: price above/below weekly EMA21
        trend_up = close[i] > ema21_1w_aligned[i]
        trend_down = close[i] < ema21_1w_aligned[i]
        
        # Mean reversion signals: Bollinger Band touches
        bb_touch_upper = close[i] >= upper_bb[i]
        bb_touch_lower = close[i] <= lower_bb[i]
        
        # Entry conditions:
        # Long: Price touches lower BB in uptrend with elevated volatility
        # Short: Price touches upper BB in downtrend with elevated volatility
        long_entry = bb_touch_lower and trend_up and vol_filter
        short_entry = bb_touch_upper and trend_down and vol_filter
        
        # Exit conditions: Return to middle (SMA20) or opposite band touch
        long_exit = close[i] >= sma20[i] or bb_touch_upper
        short_exit = close[i] <= sma20[i] or bb_touch_lower
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_BB_Touch_Volatility_Trend_Filter"
timeframe = "4h"
leverage = 1.0