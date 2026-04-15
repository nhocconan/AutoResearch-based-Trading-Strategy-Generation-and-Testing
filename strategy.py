#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly VWAP Reversion + Volume Spike + ADX Filter
# Long when price > weekly VWAP and price < 1d Bollinger Lower Band (2,2) with volume spike and ADX < 20 (range).
# Short when price < weekly VWAP and price > 1d Bollinger Upper Band (2,2) with volume spike and ADX < 20.
# Uses mean reversion in ranging markets with volume confirmation to avoid false signals.
# Weekly VWAP provides institutional reference; Bollinger Bands identify overextension.
# ADX < 20 ensures ranging conditions; volume spike confirms institutional interest.
# Discrete sizing (0.25) limits overtrading. Target: 15-25 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly VWAP (typical price * volume cumulative)
    df_1w = get_htf_data(prices, '1w')
    typical_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vwap_num = np.cumsum(typical_1w * df_1w['volume'].values)
    vwap_den = np.cumsum(df_1w['volume'].values)
    vwap_1w = vwap_num / vwap_den
    vwap_1w = np.where(vwap_den == 0, np.nan, vwap_1w)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # 1-day Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    
    # ADX (14) for ranging market filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price > weekly VWAP, price < lower BB, volume spike, ADX < 20 (ranging)
        if (close[i] > vwap_1w_aligned[i] and close[i] < lower[i] and 
            volume[i] > vol_threshold[i] and adx[i] < 20):
            signals[i] = 0.25
        
        # Short: price < weekly VWAP, price > upper BB, volume spike, ADX < 20 (ranging)
        elif (close[i] < vwap_1w_aligned[i] and close[i] > upper[i] and 
              volume[i] > vol_threshold[i] and adx[i] < 20):
            signals[i] = -0.25
        
        # Exit: ADX increases (trending) or price returns to VWAP
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (adx[i] >= 20 or close[i] >= vwap_1w_aligned[i])) or
               (signals[i-1] == -0.25 and (adx[i] >= 20 or close[i] <= vwap_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyVWAP_BB_MeanReversion"
timeframe = "1d"
leverage = 1.0