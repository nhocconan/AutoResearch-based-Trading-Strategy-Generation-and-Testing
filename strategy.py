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
    
    # Daily EMA34 trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily ATR(14) for volatility filter and stop loss
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
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_daily = ema34_daily_aligned[i]
        atr_daily = atr_daily_aligned[i]
        vol_ma_daily = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only long in daily uptrend, only short in daily downtrend
        daily_uptrend = price > ema34_daily
        daily_downtrend = price < ema34_daily
        
        # Volatility filter: avoid low volatility periods
        vol_filter_ok = atr_daily > 0
        
        # Volume filter: current volume > 1.5x daily average
        vol_ok = vol_current > 1.5 * vol_ma_daily
        
        if position == 0:
            # Long: price above daily EMA34 with volume and volatility
            if daily_uptrend and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below daily EMA34 with volume and volatility
            elif daily_downtrend and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below daily EMA34 OR volatility drops OR stop loss hit
            if not daily_uptrend or not vol_filter_ok or price < entry_price - 2.0 * atr_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA34 OR volatility drops OR stop loss hit
            if not daily_downtrend or not vol_filter_ok or price > entry_price + 2.0 * atr_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_EMA34_Trend_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0