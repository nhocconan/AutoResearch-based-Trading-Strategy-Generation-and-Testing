#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_engulfing_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 10-period ATR for volatility filter
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma  # Avoid choppy markets
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close > open_) & (open_ < close.shift(1)) & (close > close.shift(1)) & (open_ < open_.shift(1))
    # Bearish engulfing: current red candle fully engulfs previous green candle
    bearish_engulf = (close < open_) & (open_ > close.shift(1)) & (close < close.shift(1)) & (open_ > open_.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(vol_filter[i]) or
            np.isnan(bullish_engulf[i]) or np.isnan(bearish_engulf[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below EMA50 or bearish engulf
            if close[i] < ema_50_1d_aligned[i] or bearish_engulf[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above EMA50 or bullish engulf
            if close[i] > ema_50_1d_aligned[i] or bullish_engulf[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: bullish engulf + uptrend + volume spike + vol filter
            if (bullish_engulf[i] and 
                uptrend and 
                vol_spike[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish engulf + downtrend + volume spike + vol filter
            elif (bearish_engulf[i] and 
                  downtrend and 
                  vol_spike[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals