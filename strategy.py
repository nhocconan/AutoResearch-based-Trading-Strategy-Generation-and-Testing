#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ATR-based position sizing.
# Uses 1w Donchian(20) for breakout signals, filtered by 1d EMA200 for trend bias and volume spike for momentum confirmation.
# ATR-based position sizing (0.25 at low vol, 0.15 at high vol) adapts to market conditions.
# Session filter (08-20 UTC) reduces noise trades. Designed for low trade frequency (10-25/year) to minimize fee drag.
# Works in bull/bear: 1d EMA200 avoids counter-trend trades, Donchian breakouts capture sustained momentum with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: EMA200 for trend bias ===
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1w Indicators: Donchian Channel (20) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Donchian upper = max(high, lookback=20)
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low, lookback=20)
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    
    # === ATR(14) for volatility-based position sizing ===
    # True Range calculation
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile for volatility regime (using 50-period lookback)
    atr_percentile = pd.Series(atr_14).rolling(window=50, min_periods=30).rank(pct=True).values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Base position size
        base_size = 0.25
        
        # Adjust size based on volatility regime (lower size in high vol)
        if atr_percentile[i] > 0.8:  # High volatility
            size = base_size * 0.6  # Reduce to 0.15
        elif atr_percentile[i] < 0.2:  # Low volatility
            size = base_size * 1.2  # Increase to 0.30 (capped at 0.35)
        else:
            size = base_size
        
        # Cap size at 0.35
        size = min(size, 0.35)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1w Donchian upper (20-period high)
        # 2. 1d price above EMA200 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > donch_high_aligned[i] and
            close[i] > ema_200_1d_aligned[i] and
            vol_confirm):
            signals[i] = size
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1w Donchian lower (20-period low)
        # 2. 1d price below EMA200 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < donch_low_aligned[i] and
              close[i] < ema_200_1d_aligned[i] and
              vol_confirm):
            signals[i] = -size
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_1w_Donchian20_1d_EMA200_VolFilter_v1"
timeframe = "1d"
leverage = 1.0