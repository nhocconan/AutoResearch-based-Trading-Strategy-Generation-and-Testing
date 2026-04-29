#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI + 12h Supertrend + 1d ADX Regime Filter
# Long when: VW-RSI < 30 (oversold) AND 12h Supertrend = uptrend AND 1d ADX > 25 (trending market)
# Short when: VW-RSI > 70 (overbought) AND 12h Supertrend = downtrend AND 1d ADX > 25 (trending market)
# Uses VW-RSI for mean reversion in trends, Supertrend for trend direction, ADX to avoid ranging markets.
# Works in bull/bear via trend filter (Supertrend) + volatility filter (ADX) to catch pullbacks in strong trends.
# Timeframe: 6h (primary), HTF: 12h for Supertrend, 1d for ADX.

name = "6h_VolWeightedRSI_Supertrend_ADX_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2.0
    upper_basic = hl2 + (3.0 * atr_12h)
    lower_basic = hl2 - (3.0 * atr_12h)
    
    # Final Upper and Lower Bands
    final_upper = np.zeros_like(close_12h)
    final_lower = np.zeros_like(close_12h)
    supertrend = np.zeros_like(close_12h)
    trend = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_12h)):
        if i == 0:
            final_upper[i] = upper_basic[i]
            final_lower[i] = lower_basic[i]
            supertrend[i] = final_upper[i]
            trend[i] = 1
        else:
            if close_12h[i-1] > final_upper[i-1]:
                final_upper[i] = max(upper_basic[i], final_upper[i-1])
            else:
                final_upper[i] = upper_basic[i]
                
            if close_12h[i-1] < final_lower[i-1]:
                final_lower[i] = min(lower_basic[i], final_lower[i-1])
            else:
                final_lower[i] = lower_basic[i]
            
            if trend[i-1] == -1 and close_12h[i] > final_upper[i]:
                trend[i] = 1
            elif trend[i-1] == 1 and close_12h[i] < final_lower[i]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            
            supertrend[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
    
    # Align Supertrend and trend to 6h
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_12h, trend)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Volume-Weighted RSI (14-period)
    # Typical Price
    tp = (high + low + close) / 3.0
    
    # Volume-weighted typical price change
    vtp = tp * volume
    
    # Changes in VWTP
    delta = np.diff(vtp, prepend=vtp[0])
    
    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gains and losses (volume-weighted)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # RS and RSI
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(trend_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
            
        curr_rsi = rsi[i]
        curr_trend = trend_aligned[i]
        curr_adx = adx_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. VW-RSI > 70 (overbought)
            # 2. 12h trend turns down
            # 3. ADX < 20 (losing trend strength)
            if (curr_rsi > 70 or
                curr_trend == -1 or
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. VW-RSI < 30 (oversold)
            # 2. 12h trend turns up
            # 3. ADX < 20 (losing trend strength)
            if (curr_rsi < 30 or
                curr_trend == 1 or
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: VW-RSI < 30 (oversold) AND 12h uptrend AND ADX > 25 (strong trend)
            if (curr_rsi < 30 and
                curr_trend == 1 and
                curr_adx > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: VW-RSI > 70 (overbought) AND 12h downtrend AND ADX > 25 (strong trend)
            elif (curr_rsi > 70 and
                  curr_trend == -1 and
                  curr_adx > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals