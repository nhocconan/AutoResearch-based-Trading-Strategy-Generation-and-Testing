#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 100:
        return np.zeros(n)
    
    # Daily RSI(14)
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ewm = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ewm = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ewm / (loss_ewm + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily EMA(50)
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Volume MA(20)
    volume_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned daily indicators
        rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi)[i]
        ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)[i]
        vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)[i]
        
        # Check for NaN values
        if (np.isnan(rsi_daily_aligned) or np.isnan(ema_50_daily_aligned) or 
            np.isnan(vol_ma_daily_aligned)):
            continue
        
        # Volume filter (current volume > 1.5x daily volume MA)
        volume_filter = volume[i] > 1.5 * vol_ma_daily_aligned
        
        if position == 0:  # No position - look for entries
            if volume_filter:
                # Long: RSI > 50 and close above daily EMA50
                if rsi_daily_aligned > 50 and close[i] > ema_50_daily_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: RSI < 50 and close below daily EMA50
                elif rsi_daily_aligned < 50 and close[i] < ema_50_daily_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when RSI < 40 or close below EMA50
            if rsi_daily_aligned < 40 or close[i] < ema_50_daily_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when RSI > 60 or close above EMA50
            if rsi_daily_aligned > 60 or close[i] > ema_50_daily_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_DailyRSI50_EMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0