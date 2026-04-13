#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d ATR (14-period) - calculated once
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR SMA (50-period) for volatility filter - calculated once
    atr_sma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Get 1d data for higher timeframe indicators (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX (14-period) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_1d[0] = high_low_1d[0]
    
    # +DI and -DI calculation
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    atr_sma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_50)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(50, 14)
    for i in range(start, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(atr_sma_50_aligned[i]) or 
            np.isnan(atr_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_val = atr_1d[i]
        adx_val = adx_aligned[i]
        vol_filter = atr_sma_50_aligned[i]
        
        if position == 0:
            # Strong trend filter: ADX > 25
            # Low volatility filter: ATR < 50-period ATR SMA (avoid choppy markets)
            if adx_val > 25 and atr_val < vol_filter:
                # Long: close above prior 4h high + volume confirmation
                if i >= 1 and price > high[i-1] and vol > 1.5 * np.median(volume[max(0, i-20):i]):
                    position = 1
                    signals[i] = position_size
                # Short: close below prior 4h low + volume confirmation
                elif i >= 1 and price < low[i-1] and vol > 1.5 * np.median(volume[max(0, i-20):i]):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below prior 4h low OR ADX weakens
            if price < low[i-1] or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: close above prior 4h high OR ADX weakens
            if price > high[i-1] or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ADX_Trend_Filter_Vol_Breakout"
timeframe = "4h"
leverage = 1.0