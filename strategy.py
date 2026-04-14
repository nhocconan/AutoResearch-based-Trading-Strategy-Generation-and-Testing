#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    close_series = pd.Series(df_1d['close'])
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d Bollinger Band width for squeeze detection
    sma_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(bb_width_1d_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: ATR > 0.3% of price
        atr_ratio = atr_1d_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003
        
        # Bollinger Band squeeze detection: bandwidth < 4%
        bb_squeeze = bb_width_1d_aligned[i] < 0.04
        
        # Trend filter: price > 1d EMA34 for long, price < 1d EMA34 for short
        trend_filter_long = price > ema_34_1d_aligned[i]
        trend_filter_short = price < ema_34_1d_aligned[i]
        
        # Momentum filter: RSI between 35 and 65 to avoid extremes
        rsi_filter = (rsi_1d_aligned[i] > 35) & (rsi_1d_aligned[i] < 65)
        
        if position == 0:
            # Long setup: price above 1d EMA34 + volatility filter + not in squeeze + momentum filter
            if (trend_filter_long and vol_filter and not bb_squeeze and rsi_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below 1d EMA34 + volatility filter + not in squeeze + momentum filter
            elif (trend_filter_short and vol_filter and not bb_squeeze and rsi_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d EMA34
            if price < ema_34_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d EMA34
            if price > ema_34_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dEMA34_BBWidth_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0