#!/usr/bin/env python3
"""
1d_Weekly_VWAP_Reversion_WeeklyTrend
Hypothesis: Price reverting to weekly VWAP with weekly trend filter captures mean-reversion moves in ranging markets while avoiding counter-trend trades in strong trends. Weekly VWAP acts as dynamic support/resistance, and weekly trend ensures alignment with higher timeframe momentum. Works in both bull (buy dips to VWAP in uptrend) and bear (sell rallies to VWAP in downtrend) markets. Targets 7-25 trades/year on 1d to minimize fee drag.
"""

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
    
    # Get weekly data for VWAP and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP: typical price * volume cumulative / volume cumulative
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vp = typical_price * df_1w['volume']
    cum_vp = vp.cumsum()
    cum_vol = df_1w['volume'].cumsum()
    vwap = cum_vp / cum_vol
    vwap_values = vwap.values
    
    # Weekly trend: EMA50 of close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volatility filter: ATR(14) < 50th percentile of ATR(50) to avoid high volatility
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14 / atr50
    low_vol = atr_ratio < 1.0  # Below median volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ATR and EMA
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(low_vol[i])):
            signals[i] = 0.0
            continue
        
        vwap_val = vwap_aligned[i]
        ema_trend = ema50_aligned[i]
        is_low_vol = low_vol[i]
        
        if position == 0:
            # Long: price near VWAP from below in uptrend with low volatility
            if close[i] > vwap_val and close[i] < vwap_val * 1.01 and ema_trend > vwap_val and is_low_vol:
                signals[i] = size
                position = 1
            # Short: price near VWAP from above in downtrend with low volatility
            elif close[i] < vwap_val and close[i] > vwap_val * 0.99 and ema_trend < vwap_val and is_low_vol:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price moves significantly above VWAP or trend weakens
            if close[i] > vwap_val * 1.02 or ema_trend < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price moves significantly below VWAP or trend weakens
            if close[i] < vwap_val * 0.98 or ema_trend > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_VWAP_Reversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0