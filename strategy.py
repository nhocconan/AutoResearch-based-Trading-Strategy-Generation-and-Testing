#!/usr/bin/env python3
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
    
    # Get daily data for trend and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR for volatility filter
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # 4h Bollinger Bands for mean reversion signals
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(sma20[i]) or np.isnan(std20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade with daily trend
        trend_up = ema34_1d_aligned[i] > ema34_1d_aligned[i-1] if i > 0 else False
        trend_down = ema34_1d_aligned[i] < ema34_1d_aligned[i-1] if i > 0 else False
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr14_aligned[i] > np.mean(atr14_aligned[max(0, i-50):i+1]) if i >= 50 else False
        
        # Mean reversion signals from 4h Bollinger Bands
        bb_lower_touch = close[i] <= lower_bb[i]
        bb_upper_touch = close[i] >= upper_bb[i]
        
        # Entry conditions
        # Long: touch lower BB in uptrend with elevated volatility
        long_entry = bb_lower_touch and trend_up and vol_filter
        # Short: touch upper BB in downtrend with elevated volatility
        short_entry = bb_upper_touch and trend_down and vol_filter
        
        # Exit conditions: return to middle Bollinger Band
        long_exit = close[i] >= sma20[i] and position == 1
        short_exit = close[i] <= sma20[i] and position == -1
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
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

name = "4h_BollingerMeanReversion_1dTrendVolFilter"
timeframe = "4h"
leverage = 1.0