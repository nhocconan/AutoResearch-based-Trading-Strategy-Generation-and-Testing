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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for trend strength filter
    plus_dm = high_series.diff()
    minus_dm = low_series.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr_14 = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / tr_14)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / tr_14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d.values)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_14_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        # Trend strength filter: ADX > 25 for trending market
        trend_filter = adx_aligned[i] > 25
        
        # Bollinger Band squeeze detection: bandwidth < 4% for low volatility regime
        bb_squeeze = bb_width_aligned[i] < 0.04
        
        if position == 0:
            # Long setup: price touches lower BB + volatility filter + trend filter + not in squeeze
            if (price <= lower_bb_aligned[i] and vol_filter and trend_filter and not bb_squeeze):
                position = 1
                signals[i] = position_size
            # Short setup: price touches upper BB + volatility filter + trend filter + not in squeeze
            elif (price >= upper_bb_aligned[i] and vol_filter and trend_filter and not bb_squeeze):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above middle BB (SMA) or stop loss
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price >= middle_bb:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below middle BB (SMA) or stop loss
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price <= middle_bb:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_Band_Touch_ADX_Filter"
timeframe = "4h"
leverage = 1.0