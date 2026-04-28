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
    
    # Get weekly data once for long-term trend context
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly SMA(20) for trend filter
    sma_20_weekly = pd.Series(df_weekly['close'].values).rolling(window=20, min_periods=20).mean().values
    sma_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma_20_weekly)
    
    # Calculate daily ATR(14) for volatility filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_14_daily)
    
    # Calculate daily ADX(14) for trend strength
    plus_dm = np.diff(high_daily, prepend=high_daily[0])
    minus_dm = np.diff(low_daily, prepend=low_daily[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr_daily = tr  # Already calculated above
    atr_14_daily_for_adx = pd.Series(tr_daily).rolling(window=14, min_periods=14).mean().values
    
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14_daily_for_adx
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14_daily_for_adx
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_daily, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_20_weekly_aligned[i]) or np.isnan(atr_14_daily_aligned[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly SMA20
        trend_up = close[i] > sma_20_weekly_aligned[i]
        trend_down = close[i] < sma_20_weekly_aligned[i]
        
        # Volatility filter: ATR > 50th percentile of recent ATR (avoid low volatility)
        if i >= 20:
            atr_recent = atr_14_daily_aligned[max(0, i-20):i+1]
            atr_median = np.median(atr_recent[~np.isnan(atr_recent)])
            vol_filter = atr_14_daily_aligned[i] > atr_median
        else:
            vol_filter = True
        
        # Trend strength filter: ADX > 20 (trending market)
        trend_strength = adx_14_aligned[i] > 20
        
        # Entry conditions - only in strong trends with sufficient volatility
        long_entry = trend_up and vol_filter and trend_strength
        short_entry = trend_down and vol_filter and trend_strength
        
        # Exit conditions - reverse when trend changes or volatility drops
        long_exit = not trend_up or not vol_filter or not trend_strength
        short_exit = not trend_down or not vol_filter or not trend_strength
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklySMA20_Trend_Filter"
timeframe = "1d"
leverage = 1.0