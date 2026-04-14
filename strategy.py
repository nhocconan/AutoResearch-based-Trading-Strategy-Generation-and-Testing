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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 1w ATR(14) for volatility filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price > 1w EMA21 for long, price < 1w EMA21 for short
        trend_filter_long = price > ema_21_1w_aligned[i]
        trend_filter_short = price < ema_21_1w_aligned[i]
        
        # Volatility filter: ATR > 0.5% of price to avoid choppy markets
        vol_filter = atr_14_1w_aligned[i] > (0.005 * price)
        
        if position == 0:
            # Long setup: price above 1w EMA21 + volatility filter
            if trend_filter_long and vol_filter:
                position = 1
                signals[i] = position_size
            # Short setup: price below 1w EMA21 + volatility filter
            elif trend_filter_short and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1w EMA21 OR ATR drops below 0.3% of price
            if price < ema_21_1w_aligned[i] or atr_14_1w_aligned[i] < (0.003 * price):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1w EMA21 OR ATR drops below 0.3% of price
            if price > ema_21_1w_aligned[i] or atr_14_1w_aligned[i] < (0.003 * price):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wEMA21_ATR_Filter_v1"
timeframe = "1d"
leverage = 1.0