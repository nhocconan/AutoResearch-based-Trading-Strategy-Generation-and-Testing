#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data once for trend and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily close for trend and ATR
    close_daily = df_daily['close'].values
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Daily EMA34 for trend
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily ATR (14) for volatility filter
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily EMA34 of ATR for volatility regime
    atr_ema_daily = pd.Series(atr_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    atr_ema_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_ema_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(atr_ema_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_daily = ema34_daily_aligned[i]
        atr_daily = atr_daily_aligned[i]
        atr_ema_daily = atr_ema_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only long in daily uptrend, only short in daily downtrend
        daily_uptrend = price > ema34_daily
        daily_downtrend = price < ema34_daily
        
        # Volatility filter: avoid extremely low volatility (chop) and extreme volatility (panic)
        vol_normal = atr_daily > 0.5 * atr_ema_daily  # not too low
        vol_not_extreme = atr_daily < 3.0 * atr_ema_daily  # not panic spike
        
        # Volume filter: current volume > average volume (use proxy from price range)
        price_range = high[i] - low[i]
        avg_range = np.mean(np.maximum(high[max(0,i-20):i+1] - low[max(0,i-20):i+1])) if i >= 20 else price_range
        vol_ok = price_range > 0.8 * avg_range  # above average volatility in price
        
        if position == 0:
            # Long: price above daily EMA34 with normal volatility and volume
            if daily_uptrend and vol_normal and vol_not_extreme and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 with normal volatility and volume
            elif daily_downtrend and vol_normal and vol_not_extreme and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA34 OR volatility becomes extreme
            if not daily_uptrend or not vol_not_extreme:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA34 OR volatility becomes extreme
            if not daily_downtrend or not vol_not_extreme:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_EMA34_Trend_VolumeRegime_v1"
timeframe = "12h"
leverage = 1.0