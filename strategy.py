#!/usr/bin/env python3
"""
1d_KAMA_Regime_Volume_Confirmation
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) trend direction combined with
choppiness regime filter and volume confirmation to capture medium-term swings while avoiding choppy markets.
KAMA adapts to market noise, making it effective in both trending and ranging conditions.
Volume confirmation ensures breakouts have conviction. Choppiness filter avoids false signals in high-noise regimes.
Target: 15-25 trades/year per symbol with discrete sizing (0.25) to control fee drawdown.
Works in bull via trend following, in bear via regime-aware mean reversion at extremes.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array lengths: change has n-10 elements, volatility has n-1 elements
    # We need to align them properly for ER calculation
    er = np.full(n, np.nan)
    for i in range(10, n):
        if volatility[i-10:i].sum() > 0:  # volatility from i-10 to i-1
            er[i] = change[i-10] / volatility[i-10:i].sum()
        else:
            er[i] = 0
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if range_hl[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / range_hl[i]) / np.log10(14)
        else:
            chop[i] = 50  # midpoint if no range
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for KAMA (10), CHOP (14), volume MA (20), 1w EMA (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        ranging = chop[i] > 61.8
        trending = chop[i] < 38.2
        
        if position == 0:
            # Look for entry signals
            # In trending regime: follow KAMA direction with volume confirmation
            # In ranging regime: mean revert at extremes (price > KAMA for short, price < KAMA for long)
            if trending and volume_confirm[i]:
                # Trending regime: go with KAMA slope
                kama_rising = kama[i] > kama[i-1]
                kama_falling = kama[i] < kama[i-1]
                if kama_rising and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif kama_falling and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                else:
                    signals[i] = 0.0
            elif ranging and volume_confirm[i]:
                # Ranging regime: mean reversion at KAMA extremes
                if curr_close < kama[i]:  # price below KAMA -> long (reversion to mean)
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif curr_close > kama[i]:  # price above KAMA -> short (reversion to mean)
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price crosses below KAMA (trend change) or opposite regime signal
            if curr_close < kama[i] or (ranging and curr_close > kama[i] * 1.02):  # take profit in ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price crosses above KAMA (trend change) or opposite regime signal
            if curr_close > kama[i] or (ranging and curr_close < kama[i] * 0.98):  # take profit in ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0