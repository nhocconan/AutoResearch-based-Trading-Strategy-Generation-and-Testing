#!/usr/bin/env python3
"""
4h Volume-Weighted Average Price (VWAP) Deviation with 1d ATR Filter and 1w Trend
Hypothesis: Price deviations from 4h VWAP revert to the mean when accompanied by 
low volatility (ATR contraction) and aligned with higher timeframe trend (1w EMA).
This mean-reversion strategy works in both bull and bear markets by taking 
opposite positions when price deviates significantly from VWAP during low volatility
periods, while filtering for higher timeframe trend direction to avoid counter-trend trades.
Designed for 20-40 trades/year on 4h timeframe with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Calculate cumulative sums for VWAP
    cum_vwap_num = np.nancumsum(vwap_numerator)
    cum_vwap_den = np.nancumsum(vwap_denominator)
    vwap = np.divide(cum_vwap_num, cum_vwap_den, out=np.full_like(cum_vwap_num, np.nan), where=cum_vwap_den!=0)
    
    # Calculate price deviation from VWAP as percentage
    vwap_deviation = (close - vwap) / vwap * 100.0
    
    # Get 1d data for ATR filter (volatility contraction)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for volatility filter
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range calculation
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr1[0] = high_d[0] - low_d[0]  # First period TR
    tr2[0] = np.abs(high_d[0] - close_d[0])
    tr3[0] = np.abs(low_d[0] - close_d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Volatility contraction: current ATR < 50-period MA of ATR
    vol_contraction = atr_14 < (0.8 * atr_ma_50)
    
    # Get 1w data for trend filter
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend filter
    ema_50_w = pd.Series(df_w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
    signals = np.zeros(n)
    
    start_idx = 100  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or 
            np.isnan(vwap_deviation[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_ma_50[i]) or
            np.isnan(ema_50_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        deviation = vwap_deviation[i]
        ema_trend = ema_50_w_aligned[i]
        
        # Entry conditions: significant VWAP deviation + low volatility + trend alignment
        if abs(deviation) > 2.0:  # More than 2% deviation from VWAP
            if vol_contraction[i]:  # Low volatility environment
                # Long when price is significantly below VWAP and above weekly EMA (uptrend)
                if deviation < -2.0 and price > ema_trend:
                    signals[i] = 0.25
                # Short when price is significantly above VWAP and below weekly EMA (downtrend)
                elif deviation > 2.0 and price < ema_trend:
                    signals[i] = -0.25
        
        # Exit when price returns to VWAP (mean reversion complete) or volatility expands
        elif abs(deviation) < 0.5:  # Back within 0.5% of VWAP
            signals[i] = 0.0
        # Also exit if volatility expands significantly (breakdown of mean reversion environment)
        elif not vol_contraction[i] and abs(deviation) > 1.0:
            signals[i] = 0.0
    
    return signals

name = "4h_VWAP_Deviation_ATR_VolContraction_1wEMA"
timeframe = "4h"
leverage = 1.0