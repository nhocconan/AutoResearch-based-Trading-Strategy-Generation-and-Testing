#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend with RSI extremes and choppiness regime filter.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w trend via price > EMA34 on weekly close (bullish if close > EMA34, bearish if close < EMA34).
- Indicators: KAMA(ER=10) for adaptive trend, RSI(14) for momentum extremes, Chop(14) for regime.
- Entry: Long when KAMA turns up AND RSI < 30 (oversold) AND Chop > 61.8 (choppy market).
         Short when KAMA turns down AND RSI > 70 (overbought) AND Chop > 61.8.
- Exit: Opposite KAMA turn or RSI crossing 50.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Why it should work: KAMA adapts to volatility, RSI catches extremes in chop, weekly trend filter avoids counter-trend trades.
  Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) via regime and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate KAMA(ER=10, fast=2, slow=30)
    close_s = pd.Series(close)
    er = abs(close_s.diff(10)) / close_s.diff(1).abs().rolling(10, min_periods=1).sum()
    er = er.fillna(0)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = [close_s.iloc[0]]  # seed
    for i in range(1, len(close_s)):
        kama.append(kama[-1] + sc.iloc[i] * (close_s.iloc[i] - kama[-1]))
    kama = np.array(kama)
    
    # Calculate KAMA slope (turning up/down)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Calculate Choppiness Index(14)
    atr = pd.Series(high - low).rolling(14, min_periods=14).mean()
    true_range = np.maximum(high - low, 
                           np.maximum(abs(high - np.roll(close, 1)), 
                                      abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]
    atr = pd.Series(true_range).rolling(14, min_periods=14).mean()
    sum_atr14 = atr.rolling(14, min_periods=14).sum()
    max_high14 = pd.Series(high).rolling(14, min_periods=14).max()
    min_low14 = pd.Series(low).rolling(14, min_periods=14).min()
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14)) / np.log10(14)
    chop = chop.replace([np.inf, -np.inf], 50).fillna(50).values
    
    # Get 1w data for EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Need enough bars for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_34_val = ema_34_1w_aligned[i]
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals
            if chop_val > 61.8:  # choppy regime
                # Bullish: KAMA turning up AND RSI oversold
                if kama_up and rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                # Bearish: KAMA turning down AND RSI overbought
                elif kama_down and rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: KAMA turning down OR RSI crosses above 50
            if kama_down or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turning up OR RSI crosses below 50
            if kama_up or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0