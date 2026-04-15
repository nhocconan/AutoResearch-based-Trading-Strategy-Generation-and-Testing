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
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Trend and Structure ===
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h Donchian(20) for breakout structure
    donchian_high_20_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20_4h)
    donchian_low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20_4h)
    
    # === 1d Indicators: Higher Timeframe Bias ===
    # 1d EMA(100) for long-term trend
    ema_100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 1d RSI(14) for overbought/oversold filter (avoid extremes)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(donchian_high_20_4h_aligned[i]) or 
            np.isnan(donchian_low_20_4h_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1h volume > 1.5x 4h average volume per bar
        # Approximate 4h average volume per 1h bar: 4h volume SMA / 4
        vol_sma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_20_4h)
        vol_threshold = vol_sma_20_4h_aligned[i] / 4.0 * 1.5  # 1.5x average hourly volume
        vol_confirm = volume[i] > vol_threshold
        
        # === LONG CONDITIONS ===
        # 1. 4h price above EMA50 (bullish 4h trend)
        # 2. 1d price above EMA100 (bullish long-term trend)
        # 3. 1d RSI between 30 and 70 (not extreme)
        # 4. Price breaks above 4h Donchian high (breakout)
        # 5. Volume confirmation
        if (close[i] > ema_50_4h_aligned[i] and
            close[i] > ema_100_1d_aligned[i] and
            30 < rsi_14_1d_aligned[i] < 70 and
            close[i] > donchian_high_20_4h_aligned[i] and
            vol_confirm):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 4h price below EMA50 (bearish 4h trend)
        # 2. 1d price below EMA100 (bearish long-term trend)
        # 3. 1d RSI between 30 and 70 (not extreme)
        # 4. Price breaks below 4h Donchian low (breakdown)
        # 5. Volume confirmation
        elif (close[i] < ema_50_4h_aligned[i] and
              close[i] < ema_100_1d_aligned[i] and
              30 < rsi_14_1d_aligned[i] < 70 and
              close[i] < donchian_low_20_4h_aligned[i] and
              vol_confirm):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA50_EMA100_Donchian20_RSI_VolFilter_v1"
timeframe = "1h"
leverage = 1.0