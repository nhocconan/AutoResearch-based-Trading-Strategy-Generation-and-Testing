#!/usr/bin/env python3
# 1h_OrderFlow_Imbalance_Trend_Filter
# Hypothesis: 1h order flow imbalance (buy/sell volume delta) combined with 4h trend (EMA21) and 1d volatility filter
# Order flow imbalance detects institutional buying/selling pressure
# 4h EMA21 provides trend direction for higher timeframe bias
# 1d ATR percentile filter avoids trading in extreme volatility (whipsaw zones)
# Designed for 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
# Works in bull/bear via trend filter and volatility regime adaptation

name = "1h_OrderFlow_Imbalance_Trend_Filter"
timeframe = "1h"
leverage = 1.0

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
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # Order flow imbalance: buying pressure minus selling pressure
    # Positive = net buying pressure, Negative = net selling pressure
    sell_volume = volume - taker_buy_volume
    ofi = taker_buy_volume - sell_volume  # Same as 2*taker_buy_volume - volume
    
    # Normalize OFI by volume to get -1 to +1 range
    ofi_normalized = np.where(volume > 0, ofi / volume, 0)
    
    # Smooth OFI to reduce noise
    ofi_smooth = pd.Series(ofi_normalized).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 4h EMA21 for trend direction (higher timeframe bias)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr.iloc[0] = tr1.iloc[0]  # First period
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # ATR percentile rank (20-period) to identify extreme volatility
    # Avoid trading when ATR is in top/bottom 10% (extreme volatility regimes)
    atr_series = pd.Series(atr_14_1d_aligned)
    atr_rank = atr_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan, raw=False
    ).values
    
    # Session filter: 08-20 UTC (reduces noise from Asian close/US open volatility)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ofi_smooth[i]) or np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_rank[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is in middle 80% (avoid extremes)
        vol_filter = (atr_rank[i] >= 0.1) & (atr_rank[i] <= 0.9)
        
        # Trend filter: price relative to 4h EMA21
        uptrend = close[i] > ema_21_4h_aligned[i]
        downtrend = close[i] < ema_21_4h_aligned[i]
        
        if position == 0:
            # Long: strong buying pressure + uptrend + acceptable volatility
            if (ofi_smooth[i] > 0.15 and  # Significant buying pressure
                uptrend and 
                vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: strong selling pressure + downtrend + acceptable volatility
            elif (ofi_smooth[i] < -0.15 and  # Significant selling pressure
                  downtrend and 
                  vol_filter):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if selling pressure emerges or trend breaks
            if (ofi_smooth[i] < -0.05) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if buying pressure emerges or trend breaks
            if (ofi_smooth[i] > 0.05) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals