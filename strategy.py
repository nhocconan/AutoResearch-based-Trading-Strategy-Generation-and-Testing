#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: On 1h timeframe, enter long when price breaks above Camarilla R3 level AND 4h trend is up (close > EMA50) AND volume > 2.0x 20-period average. Enter short when price breaks below S3 level AND 4h trend is down (close < EMA50) AND volume spike. Uses 4h for signal direction (reduces trades) and 1h for precise entry timing. Session filter (08-20 UTC) avoids low-liquidity hours. Discrete position size 0.20 minimizes fee churn. Designed for 15-37 trades/year to avoid fee drag while capturing strong trends in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Daily Camarilla Pivot Levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + ((high-low)*1.1/4), S3 = close - ((high-low)*1.1/4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (50), volume MA warmup (20), ATR warmup (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # ATR filter: only trade when volatility is normal (not extreme)
        atr_ratio = atr[i] / np.maximum(np.mean(atr[max(0, i-50):i]), 1e-10)
        volatility_normal = (atr_ratio > 0.5) & (atr_ratio < 2.0)
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # 4h trend filter
        trend_uptrend = close[i] > ema_50_4h_aligned[i]
        trend_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price above R3 + 4h uptrend + volume spike + normal volatility + session
            long_signal = breakout_above_r3 and trend_uptrend and volume_spike[i] and volatility_normal
            
            # Short: price below S3 + 4h downtrend + volume spike + normal volatility + session
            short_signal = breakout_below_s3 and trend_downtrend and volume_spike[i] and volatility_normal
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S3 OR trend change to downtrend OR volatility extreme OR outside session
            if (close[i] < camarilla_s3_aligned[i] or not trend_uptrend or not volatility_normal or not in_session[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R3 OR trend change to uptrend OR volatility extreme OR outside session
            if (close[i] > camarilla_r3_aligned[i] or not trend_downtrend or not volatility_normal or not in_session[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0