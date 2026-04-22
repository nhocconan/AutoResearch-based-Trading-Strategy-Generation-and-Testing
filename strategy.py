#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for Pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (R1, S1, PP)
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 4-hour RSI (14-period) for momentum filter
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        atr = atr_aligned[i]
        rsi_val = rsi[i]
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr > 0.5 * atr_ma_20
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = (rsi_val > 30) and (rsi_val < 70)
        
        if position == 0:
            # Long conditions: price breaks above R1 + volume spike + volatility + RSI filter
            if price > r1 and vol_spike and vol_filter and rsi_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + volume spike + volatility + RSI filter
            elif price < s1 and vol_spike and vol_filter and rsi_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through PP or volatility drops significantly
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below PP or volatility drops
                if price < pp or atr < 0.3 * atr_ma_20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above PP or volatility drops
                if price > pp or atr < 0.3 * atr_ma_20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_ATR_Volume_RSI"
timeframe = "4h"
leverage = 1.0