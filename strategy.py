#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for trend and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
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
    
    # Daily EMA(34) for trend filter
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Load weekly data for regime filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Weekly EMA(20) for regime
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # 12h timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(ema20_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_daily = ema34_daily_aligned[i]
        atr_daily = atr_daily_aligned[i]
        ema20_weekly = ema20_weekly_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only long in daily uptrend, only short in daily downtrend
        daily_uptrend = price > ema34_daily
        daily_downtrend = price < ema34_daily
        
        # Regime filter: avoid ranging markets (price near weekly EMA)
        # Trending regime: price away from weekly EMA
        dist_from_weekly = abs(price - ema20_weekly) / ema20_weekly
        trending_regime = dist_from_weekly > 0.02  # 2% away from weekly EMA
        
        # Volume filter: current volume > 1.3x 20-period average (approximated)
        # Using current bar vs recent average
        vol_ma_recent = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.3 * vol_ma_recent
        
        if position == 0:
            # Long: price above daily EMA34, trending regime, volume confirmation
            if daily_uptrend and trending_regime and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34, trending regime, volume confirmation
            elif daily_downtrend and trending_regime and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA34 OR regime turns ranging
            if not daily_uptrend or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA34 OR regime turns ranging
            if not daily_downtrend or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_1w_EMA34_Trend_Regime_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0