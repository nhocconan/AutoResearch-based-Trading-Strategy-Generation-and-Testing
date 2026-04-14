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
    
    # Calculate 1d Close for price action
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for volatility and stop
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    close_series = pd.Series(close_1d)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for trend strength
    plus_dm = high_series.diff()
    minus_dm = low_series.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr_14 = tr.rolling(window=14, min_periods=14).sum()
    plus_di_14 = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    minus_di_14 = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    dx = (abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    adx_1d = dx.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d RSI(14) for momentum
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Align 1d indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_1d_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Trend strength filter: ADX > 25 for trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        rsi_filter = (rsi_1d_aligned[i] > 30) & (rsi_1d_aligned[i] < 70)
        
        if position == 0:
            # Entry conditions: volatility + trend strength + momentum
            if vol_filter and trend_filter and rsi_filter:
                # Simple momentum-based entry: long if RSI > 50, short if RSI < 50
                if rsi_1d_aligned[i] > 50:
                    position = 1
                    signals[i] = position_size
                elif rsi_1d_aligned[i] < 50:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: RSI crosses below 50 or volatility drops
            if (rsi_1d_aligned[i] < 50) or (atr_ratio < 0.002):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: RSI crosses above 50 or volatility drops
            if (rsi_1d_aligned[i] > 50) or (atr_ratio < 0.002):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_ADX_RSI_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0