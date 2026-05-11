#!/usr/bin/env python3
name = "6h_ChoppinessIndex_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Choppiness Index on 6h (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - min(low) over 14 periods
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_h - min_l
    
    # Choppiness Index
    chop = np.zeros_like(close)
    chop[:] = np.nan
    valid = (tr_sum > 0) & (range_hl > 0)
    chop[valid] = 100 * np.log10(tr_sum[valid] / range_hl[valid]) / np.log10(14)
    
    # Choppiness regime thresholds
    chop_high = 61.8  # > 61.8 = ranging (mean revert)
    chop_low = 38.2   # < 38.2 = trending (trend follow)
    
    # Mean reversion signals (in ranging markets)
    # RSI(14) for mean reversion
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Trend following signals (in trending markets)
    # EMA crossover (8,21)
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_cross_up = ema8 > ema21
    ema_cross_down = ema8 < ema21
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 14)  # Need enough data for EMA21 and chop
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(ema8[i]) or 
            np.isnan(ema21[i]) or np.isnan(trend_up_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        chop_val = chop[i]
        rsi_val = rsi[i]
        ema8_val = ema8[i]
        ema21_val = ema21[i]
        trend_up = trend_up_1d_aligned[i]
        
        if position == 0:
            # In ranging market (chop > 61.8): mean reversion at RSI extremes
            if chop_val > chop_high:
                if rsi_val < 30:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:  # Overbought
                    signals[i] = -0.25
                    position = -1
            # In trending market (chop < 38.2): trend following with daily filter
            elif chop_val < chop_low:
                if ema8_val > ema21_val and trend_up:  # Uptrend + daily up
                    signals[i] = 0.25
                    position = 1
                elif ema8_val < ema21_val and not trend_up:  # Downtrend + daily down
                    signals[i] = -0.25
                    position = -1
            # In transition zone (38.2 <= chop <= 61.8): no trade
        
        elif position == 1:
            # Long exit conditions
            if chop_val > chop_high:  # Went to ranging: exit on RSI > 50
                if rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop_val < chop_low:  # Still trending: exit on EMA cross down
                if ema8_val < ema21_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Transition zone: exit on RSI > 50
                if rsi_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if chop_val > chop_high:  # Went to ranging: exit on RSI < 50
                if rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop_val < chop_low:  # Still trending: exit on EMA cross up
                if ema8_val > ema21_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Transition zone: exit on RSI < 50
                if rsi_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals