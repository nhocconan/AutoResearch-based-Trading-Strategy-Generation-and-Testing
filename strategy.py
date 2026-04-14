#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for volatility
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d ADX(14) for trend strength
    plus_dm = high_1d.diff()
    minus_dm = low_1d.shift(1) - low_1d
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr_14 = tr_1d.rolling(window=14, min_periods=14).sum()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: ATR > 0.5% of price
        vol_filter = atr_1d_aligned[i] > (price * 0.005) if price > 0 else False
        
        # Trend strength filter: ADX > 25
        trend_strength = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x average
        vol_confirm = vol > (vol_avg[i] * 1.3) if not np.isnan(vol_avg[i]) else False
        
        if position == 0:
            # Long: price above EMA50 + volatility + trend strength + volume confirmation
            if (price > ema_50_1d_aligned[i] and vol_filter and trend_strength and vol_confirm):
                position = 1
                signals[i] = position_size
            # Short: price below EMA50 + volatility + trend strength + volume confirmation
            elif (price < ema_50_1d_aligned[i] and vol_filter and trend_strength and vol_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50 OR ADX < 20 (trend weakening)
            if price < ema_50_1d_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA50 OR ADX < 20 (trend weakening)
            if price > ema_50_1d_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dADX_EMA50_Volume_Filter"
timeframe = "4h"
leverage = 1.0