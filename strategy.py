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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX(14) for trend strength
    high_1w = pd.Series(df_1w['high'])
    low_1w = pd.Series(df_1w['low'])
    close_1w = pd.Series(df_1w['close'])
    
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_1w.diff()
    minus_dm = low_1w.diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr_14 = tr_1w.rolling(window=14, min_periods=14).sum()
    plus_di_14 = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    minus_di_14 = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    dx = (abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    adx_14 = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx_14.values
    
    # Calculate weekly Donchian Channel (20)
    dc_high_20 = high_1w.rolling(window=20, min_periods=20).max().values
    dc_low_20 = low_1w.rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    dc_high_aligned = align_htf_to_ltf(prices, df_1w, dc_high_20)
    dc_low_aligned = align_htf_to_ltf(prices, df_1w, dc_low_20)
    
    # Calculate daily ATR(14) for volatility filter and position sizing
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr1_d[0] = 0
    tr2_d[0] = 0
    tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(dc_high_aligned[i]) or 
            np.isnan(dc_low_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        trend_filter = adx_aligned[i] > 25
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + trend + vol
            if (price > dc_high_aligned[i] and trend_filter and vol_filter):
                position = 1
                signals[i] = position_size
            # Short entry: price breaks below weekly Donchian low + trend + vol
            elif (price < dc_low_aligned[i] and trend_filter and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low
            if price < dc_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high
            if price > dc_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_weeklyADX_DonchianBreakout"
timeframe = "1d"
leverage = 1.0