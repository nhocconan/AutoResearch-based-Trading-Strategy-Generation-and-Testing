#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RVOL_2x_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Relative Volume (RVOL): current volume / 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / np.where(vol_ma_20 > 0, vol_ma_20, np.nan)
    
    # Trend filter: 100-period EMA on close
    ema_100 = pd.Series(close).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load daily data for trend filter (EMA 50 on daily close)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = max(100, 50)  # EMA100 needs 100 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isclose(rvol[i], 0) or np.isnan(rvol[i]) or
            np.isnan(ema_100[i]) or np.isnan(atr_14[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions:
        # 1. High relative volume (RVOL > 2.0) - indicates institutional interest
        # 2. Price above/below 100 EMA for trend direction
        # 3. Daily EMA50 filter to avoid counter-trend trades in strong daily trends
        # 4. ATR filter: only trade when volatility is sufficient (ATR > 0)
        
        rvol_filter = rvol[i] > 2.0
        price_above_ema = close[i] > ema_100[i]
        price_below_ema = close[i] < ema_100[i]
        daily_uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] if i > 0 else True
        daily_downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] if i > 0 else True
        vol_filter = atr_14[i] > 0
        
        # Long: High RVOL + price above 100 EMA + daily uptrend
        if rvol_filter and price_above_ema and daily_uptrend and vol_filter:
            signals[i] = 0.25
        # Short: High RVOL + price below 100 EMA + daily downtrend
        elif rvol_filter and price_below_ema and daily_downtrend and vol_filter:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals