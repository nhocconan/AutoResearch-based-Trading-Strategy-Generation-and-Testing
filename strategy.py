#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Load weekly data for additional trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly EMA34 for trend
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Daily ATR (14) for volatility filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily volume average (20) for volume confirmation
    volume_daily = df_daily['volume'].values
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
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(ema34_weekly_aligned[i]) or 
            np.isnan(atr_daily_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_daily = ema34_daily_aligned[i]
        ema34_weekly = ema34_weekly_aligned[i]
        atr_daily = atr_daily_aligned[i]
        vol_ma_daily = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only trade when both daily and weekly agree
        daily_uptrend = price > ema34_daily
        daily_downtrend = price < ema34_daily
        weekly_uptrend = price > ema34_weekly
        weekly_downtrend = price < ema34_weekly
        
        # Volatility filter: avoid low volatility periods
        vol_filter_ok = atr_daily > 0
        
        # Volume filter: current volume > 1.5x daily average
        vol_ok = vol_current > 1.5 * vol_ma_daily
        
        # Combined trend signal: both timeframes must agree
        both_uptrend = daily_uptrend and weekly_uptrend
        both_downtrend = daily_downtrend and weekly_downtrend
        
        if position == 0:
            # Long: price above both EMAs with volume and volatility
            if both_uptrend and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below both EMAs with volume and volatility
            elif both_downtrend and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below either EMA OR volatility drops
            if not both_uptrend or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above either EMA OR volatility drops
            if not both_downtrend or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_1w_EMA34_DualTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0