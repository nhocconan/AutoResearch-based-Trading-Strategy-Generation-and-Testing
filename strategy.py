#!/usr/bin/env python3
# 1d_1w_funding_zscore_mean_reversion
# Strategy: Funding rate mean reversion with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Extreme funding rates (Z-score) signal mean reversion. Weekly trend filter ensures
# trades align with higher timeframe momentum. Volume confirmation filters weak signals.
# Works in bull by fading euphoria and in bear by fading panic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_funding_zscore_mean_reversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = ema_50_1w > np.roll(ema_50_1w, 1)  # Rising EMA
    weekly_uptrend[0] = False  # First value invalid
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    # Daily ATR(14) for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Funding rate proxy using price deviation from weekly VWAP
    # Since we don't have direct funding data, use deviation from weekly VWAP as proxy
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    # Weekly VWAP calculation
    pv_sum = pd.Series(pv).rolling(window=5*7*12, min_periods=5*7*12).sum()  # Approx 5 days of 12h periods per week
    vol_sum = pd.Series(volume).rolling(window=5*7*12, min_periods=5*7*12).sum()
    weekly_vwap = pv_sum / vol_sum
    weekly_vwap = np.nan_to_num(weekly_vwap, nan=close[0])
    
    # Deviation from weekly VWAP as funding proxy
    deviation = (close - weekly_vwap) / atr
    
    # Z-score of deviation over 60-day lookback
    def rolling_zscore(arr, window):
        mean = pd.Series(arr).rolling(window=window, min_periods=window).mean()
        std = pd.Series(arr).rolling(window=window, min_periods=window).std()
        return (arr - mean) / (std + 1e-10)
    
    zscore = rolling_zscore(deviation, 60)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after zscore warmup
        # Skip if any required data is invalid
        if (np.isnan(zscore[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals with weekly trend filter
        long_signal = (zscore[i] < -2.0) and weekly_uptrend_aligned[i] and vol_spike[i]
        short_signal = (zscore[i] > 2.0) and (not weekly_uptrend_aligned[i]) and vol_spike[i]
        
        # Exit conditions: Z-score reverts to mean or opposite extreme
        exit_long = position == 1 and (zscore[i] > -0.5 or zscore[i] > 2.0)
        exit_short = position == -1 and (zscore[i] < 0.5 or zscore[i] < -2.0)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals