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
    
    # Load daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA (50-period) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12-period RSI for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Align indicators to 12-hour timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: require sufficient daily volatility
        vol_filter = atr_14_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: Price above EMA50 + RSI > 50 + volume spike + volatility filter
            if (close[i] > ema50_1d_aligned[i] and 
                rsi_aligned[i] > 50 and 
                vol_spike[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA50 + RSI < 50 + volume spike + volatility filter
            elif (close[i] < ema50_1d_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  vol_spike[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses EMA50 or RSI reaches extreme
            if position == 1:
                if (close[i] < ema50_1d_aligned[i] or 
                    rsi_aligned[i] < 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema50_1d_aligned[i] or 
                    rsi_aligned[i] > 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_EMA50_RSI_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0