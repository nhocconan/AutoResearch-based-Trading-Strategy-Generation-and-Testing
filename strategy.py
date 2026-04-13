#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align all data to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any required data is not ready
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > 20-period average (high volatility regime)
        if i >= 20:
            atr_ma_20 = np.nanmean(atr_1d_aligned[i-20:i]) if not np.isnan(atr_1d_aligned[i-20:i]).all() else np.nan
            if np.isnan(atr_ma_20) or atr_1d_aligned[i] < atr_ma_20:
                signals[i] = 0.0 if position == 0 else (position_size if position == 1 else -position_size)
                continue
        
        # Trend filter: only long when price > weekly EMA50, short when price < weekly EMA50
        long_trend = close[i] > ema_50_1w_aligned[i]
        short_trend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: volatility breakout with trend confirmation
        # Long when price breaks above recent high with volatility and uptrend
        # Short when price breaks below recent low with volatility and downtrend
        if i >= 5:
            recent_high = np.max(high[i-5:i])
            recent_low = np.min(low[i-5:i])
            
            if position == 0:
                if close[i] > recent_high and long_trend:
                    position = 1
                    signals[i] = position_size
                elif close[i] < recent_low and short_trend:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit when price closes below recent low or trend changes
                if close[i] < recent_low or not long_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit when price closes above recent high or trend changes
                if close[i] > recent_high or not short_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d1w_Volatility_Trend_Breakout"
timeframe = "4h"
leverage = 1.0