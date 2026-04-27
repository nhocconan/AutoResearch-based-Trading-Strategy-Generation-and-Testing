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
    
    # Get weekly data for trend filter and monthly data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    df_1M = get_htf_data(prices, '1M')
    
    if len(df_1w) < 2 or len(df_1M) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1M = df_1M['close'].values
    
    # Weekly EMA trend filter (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Monthly EMA momentum filter (13-period)
    ema_13_1M = pd.Series(close_1M).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1M_aligned = align_htf_to_ltf(prices, df_1M, ema_13_1M)
    
    # Daily ATR for volatility filter (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily close for price action
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (21), monthly EMA (13), daily ATR (14), daily close
    start_idx = max(21, 13, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_13_1M_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_21_1w = ema_21_1w_aligned[i]
        ema_13_1M = ema_13_1M_aligned[i]
        atr_14_1d = atr_14_1d_aligned[i]
        close_1d_val = close_1d_aligned[i]
        
        # Trend alignment: price above both weekly and monthly EMA
        bullish_alignment = price > ema_21_1w and price > ema_13_1M
        bearish_alignment = price < ema_21_1w and price < ema_13_1M
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_1d > 0.01 * close_1d_val  # ATR > 1% of price
        
        if position == 0:
            # Long: bullish alignment + volatility filter
            if bullish_alignment and vol_filter:
                signals[i] = size
                position = 1
            # Short: bearish alignment + volatility filter
            elif bearish_alignment and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or volatility collapse
            if bearish_alignment or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish alignment or volatility collapse
            if bullish_alignment or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyMonthlyEMA_Alignment_VolumeFilter"
timeframe = "1d"
leverage = 1.0