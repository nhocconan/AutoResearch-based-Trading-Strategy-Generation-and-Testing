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
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(atr_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_daily = ema34_daily_aligned[i]
        atr_daily = atr_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only long in daily uptrend, only short in daily downtrend
        daily_uptrend = price > ema34_daily
        daily_downtrend = price < ema34_daily
        
        # Volatility filter: avoid low volatility periods
        # Use ATR relative to price to normalize
        atr_ratio = atr_daily / price if price > 0 else 0
        vol_ok = atr_ratio > 0.015  # At least 1.5% volatility
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma_recent = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_spike = vol_current > 1.5 * vol_ma_recent
        
        if position == 0:
            # Long: price above daily EMA34, volatility OK, volume spike
            if daily_uptrend and vol_ok and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34, volatility OK, volume spike
            elif daily_downtrend and vol_ok and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA34 OR volatility drops
            if not daily_uptrend or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA34 OR volatility drops
            if not daily_downtrend or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_EMA34_Trend_VolVolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0