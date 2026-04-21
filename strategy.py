#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d - low_1d
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Load 4h data for entry signal and volume
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    vol_4h = df_4h['volume'].values
    
    # 4h EMA20 for entry trigger
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h volume average for spike detection
    vol_avg_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators to the 4h timeframe (no shift needed as we're already in 4h)
    ema20_4h_aligned = ema20_4h  # Already on 4h timeframe
    vol_avg_20_4h_aligned = vol_avg_20_4h  # Already on 4h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_daily = ema50_1d_aligned[i]
        atr_daily = atr14_1d_aligned[i]
        ema20_4h_val = ema20_4h_aligned[i]
        vol_avg_20 = vol_avg_20_4h_aligned[i]
        vol = vol_4h[i]
        price = close_4h[i]
        
        # Volatility filter: daily ATR > 50% of its 20-period average (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr_daily > 0.5 * atr_ma_20
        
        # Trend filter: price above/below daily EMA50
        uptrend = price > ema50_daily
        downtrend = price < ema50_daily
        
        # Volume spike detection
        vol_spike = vol > 1.5 * vol_avg_20
        
        if position == 0:
            # Long: price crosses above 4h EMA20 + daily uptrend + volatility + volume spike
            if price > ema20_4h_val and uptrend and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below 4h EMA20 + daily downtrend + volatility + volume spike
            elif price < ema20_4h_val and downtrend and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through 4h EMA20 or volatility drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below EMA20 or volatility collapse
                if price < ema20_4h_val or not vol_filter:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above EMA20 or volatility collapse
                if price > ema20_4h_val or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_EMA20_Cross_DailyTrend_VolFilter"
timeframe = "4h"
leverage = 1.0