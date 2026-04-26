#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeRegime_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volatility regime filter (ATR ratio < 1.0) to capture strong momentum moves in trending markets. Uses tighter breakout levels (R3/S3) and volatility filter to reduce false breakouts and overtrading, targeting 20-40 trades/year. Works in bull/bear markets by only trading with the 1d trend and avoiding high-volatility, choppy conditions.
"""

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
    
    # Get 1d data for HTF trend (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r3 = prev_close_1d + (1.0/4) * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - (1.0/4) * (prev_high_1d - prev_low_1d)
    
    # ATR(14) for volatility measurement
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period ATR mean (volatility regime filter)
    atr_mean = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_mean  # < 1.0 = low volatility, > 1.0 = high volatility
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(34) 1d, ATR mean (50)
    start_idx = max(34, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr_ratio[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_ratio_val = atr_ratio[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        # Volatility regime filter: only trade in low-moderate volatility (avoid chop)
        vol_regime = atr_ratio_val < 1.0
        
        if position == 0:
            # Long: break above R3 with volatility regime, and uptrend
            long_signal = (close_val > camarilla_r3_aligned[i]) and \
                          vol_regime and \
                          uptrend
            
            # Short: break below S3 with volatility regime, and downtrend
            short_signal = (close_val < camarilla_s3_aligned[i]) and \
                           vol_regime and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit on trend reversal
            if close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit on trend reversal
            if close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0