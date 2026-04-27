#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter and 1d volatility filter.
# Uses 4h EMA20 for trend direction and 1d ATR for volatility-based position sizing.
# Entry on 1h pullbacks to EMA20 during strong trends, filtered by 1d volatility regime.
# Designed to work in both bull (trend following) and bear (mean reversion within trend) markets.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    atr_14_1d = np.full_like(tr, np.nan, dtype=np.float64)
    for i in range(14, len(tr)):
        atr_14_1d[i] = np.nanmean(tr[i-13:i+1])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h EMA20 for entry timing
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # Base position size
    
    # Warmup: need 4h EMA (20), 1d ATR (14), 1h EMA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(ema_20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend_4h = ema_20_4h_aligned[i]
        ema_1h = ema_20_1h[i]
        atr_1d = atr_14_1d_aligned[i]
        
        # Volatility filter: avoid extremely low or high volatility days
        # Use 50-period median of ATR to normalize
        if i >= 50:
            atr_lookback = atr_14_1d_aligned[max(0, i-49):i+1]
            atr_median = np.nanmedian(atr_lookback)
            vol_filter = (atr_1d > 0.5 * atr_median) and (atr_1d < 2.0 * atr_median)
        else:
            vol_filter = True  # Not enough data for vol filter yet
        
        if position == 0:
            # Long: price above 4h EMA (uptrend) and pulls back to 1h EMA
            if price > ema_trend_4h and price <= ema_1h and vol_filter:
                signals[i] = base_size
                position = 1
            # Short: price below 4h EMA (downtrend) and bounces to 1h EMA
            elif price < ema_trend_4h and price >= ema_1h and vol_filter:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 1h EMA or 4h trend turns down
            if price < ema_1h or price < ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Exit short: price breaks above 1h EMA or 4h trend turns up
            if price > ema_1h or price > ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_EMA20_Trend_Pullback_VolatilityFiltered"
timeframe = "1h"
leverage = 1.0