#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX Trend + 1d RSI Mean Reversion + Volume Spike (1h timeframe)
# Uses ADX(14) on 4h to filter for trending markets (ADX > 25).
# In trending markets: long when 4h EMA(21) slope > 0 and RSI(14) < 40, short when EMA slope < 0 and RSI > 60.
# In ranging markets (ADX <= 25): long when RSI(14) < 30, short when RSI(14) > 70.
# Volume confirmation requires > 1.5x 20-bar median volume on 1h.
# Session filter: only trade 08-20 UTC to avoid low-liquidity hours.
# Designed to work in bull markets (trend following) and bear markets (mean reversion via RSI extremes).
# Conservative sizing (0.20) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h ADX(14) for trend strength
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_rolled = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_rolled = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_rolled = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Avoid division by zero
    tr_rolled = tr_rolled.replace(0, np.nan)
    di_plus = 100 * dm_plus_rolled / tr_rolled
    di_minus = 100 * dm_minus_rolled / tr_rolled
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_4h = adx.values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 4h EMA(21) for trend direction
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean()
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h.values)
    ema_slope_4h = np.diff(ema_21_4h_aligned, prepend=ema_21_4h_aligned[0])
    
    # 1-day RSI(14) for mean reversion
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN or outside session
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema_slope_4h[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_threshold[i]) or
            not in_session[i]):
            continue
        
        adx_val = adx_4h_aligned[i]
        ema_slope = ema_slope_4h[i]
        rsi = rsi_1d_aligned[i]
        vol = volume[i]
        vol_thresh = vol_threshold[i]
        
        # Trending market (ADX > 25): follow EMA slope with RSI filter
        if adx_val > 25:
            # Long: EMA upward slope, RSI oversold (<40), volume spike
            if (ema_slope > 0 and 
                rsi < 40 and 
                vol > vol_thresh):
                signals[i] = 0.20
            
            # Short: EMA downward slope, RSI overbought (>60), volume spike
            elif (ema_slope < 0 and 
                  rsi > 60 and 
                  vol > vol_thresh):
                signals[i] = -0.20
        
        # Ranging market (ADX <= 25): pure mean reversion
        else:
            # Long: RSI deeply oversold (<30), volume spike
            if (rsi < 30 and 
                vol > vol_thresh):
                signals[i] = 0.20
            
            # Short: RSI deeply overbought (>70), volume spike
            elif (rsi > 70 and 
                  vol > vol_thresh):
                signals[i] = -0.20
        
        # Exit conditions: ADX drops below 20 (trend weakening) or RSI returns to neutral range
        if i > 0 and signals[i-1] != 0:
            if (adx_4h_aligned[i] < 20 or 
                (signals[i-1] == 0.20 and rsi >= 40) or 
                (signals[i-1] == -0.20 and rsi <= 60)):
                signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_ADX_RSI1d_Volume_Session"
timeframe = "1h"
leverage = 1.0