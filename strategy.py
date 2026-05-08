#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Squeeze_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h trend: EMA(50) and EMA(200)
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 1d volatility squeeze: Bollinger Band width percentile
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20_1d + 2 * std_20_1d
    lower_bb = sma_20_1d - 2 * std_20_1d
    bb_width = (upper_bb - lower_bb) / sma_20_1d
    
    # Percentile rank of BB width (252-day lookback for stability)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    
    # 1h volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(bb_width_pct_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish trend: EMA20 > EMA50 > EMA200 + low volatility squeeze
            if (ema_20_4h_aligned[i] > ema_50_4h_aligned[i] > ema_200_4h_aligned[i] and
                bb_width_pct_aligned[i] < 0.3 and  # Low volatility (bottom 30% percentile)
                vol_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Bearish trend: EMA20 < EMA50 < EMA200 + low volatility squeeze
            elif (ema_20_4h_aligned[i] < ema_50_4h_aligned[i] < ema_200_4h_aligned[i] and
                  bb_width_pct_aligned[i] < 0.3 and  # Low volatility (bottom 30% percentile)
                  vol_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or volatility expands
            if (ema_20_4h_aligned[i] <= ema_50_4h_aligned[i] or
                bb_width_pct_aligned[i] > 0.7):  # High volatility (top 30% percentile)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or volatility expands
            if (ema_20_4h_aligned[i] >= ema_50_4h_aligned[i] or
                bb_width_pct_aligned[i] > 0.7):  # High volatility (top 30% percentile)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals