#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA(20) for trend filter
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Load daily data for volume and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    volume_daily = df_daily['volume'].values
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily ATR (14)
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily volume average (20)
    vol_ma_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema20_weekly_aligned[i]) or 
            np.isnan(atr_daily_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema20_weekly = ema20_weekly_aligned[i]
        atr_daily = atr_daily_aligned[i]
        vol_ma_daily = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only long in weekly uptrend, only short in weekly downtrend
        weekly_uptrend = price > ema20_weekly
        weekly_downtrend = price < ema20_weekly
        
        # Volatility filter: avoid low volatility periods
        atr_ratio = atr_daily / price if price > 0 else 0
        vol_ok = atr_ratio > 0.02  # At least 2% volatility
        
        # Volume filter: current volume > 2x daily average volume
        vol_spike = vol_current > 2.0 * vol_ma_daily
        
        if position == 0:
            # Long: price above weekly EMA20, volatility OK, volume spike
            if weekly_uptrend and vol_ok and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA20, volatility OK, volume spike
            elif weekly_downtrend and vol_ok and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA20 OR volatility drops OR volume dries up
            if not weekly_uptrend or not vol_ok or vol_current < 0.5 * vol_ma_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA20 OR volatility drops OR volume dries up
            if not weekly_downtrend or not vol_ok or vol_current < 0.5 * vol_ma_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_EMA20_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0