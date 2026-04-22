#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h ATR for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA for trend (fast and slow)
    ema_fast = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align daily indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_val = atr[i]
        ema_fast_val = ema_fast[i]
        ema_slow_val = ema_slow[i]
        atr_1d = atr_1d_aligned[i]
        atr_ma_1d = atr_ma_1d_aligned[i]
        
        # Volume filter: current volume above 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else 0
        vol_filter = vol > vol_ma * 1.5 if vol_ma > 0 else False
        
        # Trend filter: EMA alignment
        bullish_trend = ema_fast_val > ema_slow_val
        bearish_trend = ema_fast_val < ema_slow_val
        
        # Volatility regime: trade only when daily ATR is elevated (trending market)
        vol_regime = atr_1d > atr_ma_1d
        
        # Entry conditions
        if position == 0 and vol_regime and vol_filter:
            # Long: bullish trend + price above fast EMA + volatility expansion
            if bullish_trend and price > ema_fast_val and atr_val > atr[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: bearish trend + price below fast EMA + volatility expansion
            elif bearish_trend and price < ema_fast_val and atr_val > atr[i-1]:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position != 0:
            # Exit on trend reversal or volatility collapse
            trend_reversal = (position == 1 and ema_fast_val < ema_slow_val) or \
                            (position == -1 and ema_fast_val > ema_slow_val)
            vol_collapse = atr_val < 0.5 * atr[i-1]  # Sharp volatility drop
            
            if trend_reversal or vol_collapse:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_EMA_Volatility_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0