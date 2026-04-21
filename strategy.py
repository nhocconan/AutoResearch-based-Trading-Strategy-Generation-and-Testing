#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Load daily data for volatility and volume filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Daily volume SMA for volume filter
    vol_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # 12h timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA20 for entry signal
    ema20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(atr_10_aligned[i]) or 
            np.isnan(vol_sma_20_aligned[i]) or 
            np.isnan(ema20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        atr_10_val = atr_10_aligned[i]
        vol_sma_20_val = vol_sma_20_aligned[i]
        vol = volume[i]
        price = close[i]
        ema20_12h_val = ema20_12h[i]
        
        # Volatility filter: daily ATR > 50% of its 20-period average (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr_10_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr_10_val > 0.5 * atr_ma_20
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_filter and (vol > 1.5 * vol_sma_20_val)
        
        # Trend filter: price above/below weekly EMA34
        uptrend = price > ema34_1w_val
        downtrend = price < ema34_1w_val
        
        if position == 0:
            # Long: price above 12h EMA20 + weekly uptrend + volatility + volume filter
            if price > ema20_12h_val and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA20 + weekly downtrend + volatility + volume filter
            elif price < ema20_12h_val and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through 12h EMA20
            exit_signal = False
            
            if position == 1:  # long position
                if price < ema20_12h_val:
                    exit_signal = True
            elif position == -1:  # short position
                if price > ema20_12h_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyEMA34_Trend_EMA20_VolVolFilter"
timeframe = "12h"
leverage = 1.0