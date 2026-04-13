#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 4h EMA200 for trend filter
    ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all data to 4h timeframe
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 4-period RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi_4 = 100 - (100 / (1 + rs))
    rsi_4 = rsi_4.values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(rsi_4[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when current ATR > 50-period ATR MA (high volatility regime)
        # Approximate current ATR using recent price action
        if i >= 14:
            tr_current = np.abs(high[i] - low[i])
            tr_prev1 = np.abs(high[i] - close[i-1])
            tr_prev2 = np.abs(low[i] - close[i-1])
            tr_current = np.maximum(tr_current, np.maximum(tr_prev1, tr_prev2))
            # Simple ATR approximation for current period
            atr_current = tr_current  # Simplified - in reality would need smoothing
            volatility_filter = atr_current > atr_ma_50_aligned[i]
        else:
            volatility_filter = False
        
        # Trend filter: long when price > EMA200, short when price < EMA200
        long_trend = close[i] > ema_200_4h_aligned[i]
        short_trend = close[i] < ema_200_4h_aligned[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        momentum_filter = (rsi_4[i] > 30) and (rsi_4[i] < 70)
        
        # Entry conditions: volatility + trend + momentum
        if position == 0:
            if volatility_filter and long_trend and momentum_filter:
                position = 1
                signals[i] = position_size
            elif volatility_filter and short_trend and momentum_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when trend changes or momentum deteriorates
            if not (long_trend and momentum_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when trend changes or momentum deteriorates
            if not (short_trend and momentum_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ATR_Volatility_Trend_Momentum_Filter"
timeframe = "4h"
leverage = 1.0