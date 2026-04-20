#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for trend and regime
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Weekly ATR (14)
    tr1 = np.abs(high_weekly - low_weekly)
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr1[0] = high_weekly[0] - low_weekly[0]
    tr2[0] = np.abs(high_weekly[0] - close_weekly[0])
    tr3[0] = np.abs(low_weekly[0] - close_weekly[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    
    # Weekly EMA(34) for trend filter
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Weekly EMA(20) for regime
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Load daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    volume_daily = df_daily['volume'].values
    vol_ma20_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma20_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(atr_weekly_aligned[i]) or 
            np.isnan(ema20_weekly_aligned[i]) or np.isnan(vol_ma20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_weekly = ema34_weekly_aligned[i]
        atr_weekly = atr_weekly_aligned[i]
        ema20_weekly = ema20_weekly_aligned[i]
        vol_ma20_daily = vol_ma20_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only long in weekly uptrend, only short in weekly downtrend
        weekly_uptrend = price > ema34_weekly
        weekly_downtrend = price < ema34_weekly
        
        # Regime filter: avoid ranging markets (price near weekly EMA)
        # Trending regime: price away from weekly EMA
        dist_from_weekly = abs(price - ema20_weekly) / ema20_weekly
        trending_regime = dist_from_weekly > 0.03  # 3% away from weekly EMA
        
        # Volume filter: current volume > 1.4x daily 20-period average
        vol_ok = vol_current > 1.4 * vol_ma20_daily
        
        if position == 0:
            # Long: price above weekly EMA34, trending regime, volume confirmation
            if weekly_uptrend and trending_regime and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34, trending regime, volume confirmation
            elif weekly_downtrend and trending_regime and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA34 OR regime turns ranging
            if not weekly_uptrend or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA34 OR regime turns ranging
            if not weekly_downtrend or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_1d_EMA34_Trend_Regime_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0