#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Volume_Confirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend
    df_weekly = get_htf_data(prices, '1w')
    # Get daily data ONCE before loop for volume average
    df_daily = get_htf_data(prices, '1d')
    
    if len(df_weekly) < 2 or len(df_daily) < 2:
        return np.zeros(n)
    
    # Weekly EMA10 for trend direction
    close_weekly = df_weekly['close'].values
    ema_10_weekly = pd.Series(close_weekly).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_10_weekly)
    
    # Daily average volume (20-period) for volume confirmation
    vol_daily = df_daily['volume'].values
    vol_avg_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    vol_avg_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_daily)
    
    # Daily ATR for exit (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_10_weekly_aligned[i]
        vol_avg = vol_avg_daily_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(vol_avg) or np.isnan(current_atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5x daily average volume
        vol_spike = current_volume > 1.5 * vol_avg
        
        if position == 0:
            # Long: price above weekly EMA10 with volume spike
            if current_close > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price below weekly EMA10 with volume spike
            elif current_close < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price below weekly EMA10 or ATR stop loss
            if current_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly EMA10 or ATR stop loss
            if current_close > ema_trend:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals