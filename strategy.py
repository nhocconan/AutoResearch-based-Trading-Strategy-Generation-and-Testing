#!/usr/bin/env python3
# 1d_kama_rsi_chop_filter_v1
# Hypothesis: 1d strategy using KAMA for trend direction, RSI(14) for momentum confirmation, and Choppiness Index regime filter (CHOP>61.8 = ranging, CHOP<38.2 = trending). Uses 1w HTF EMA(34) for higher timeframe trend alignment. Discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year (30-100 total over 4 years). Works in bull/bear: KAMA adapts to trend changes, RSI avoids overextended entries, chop filter ensures trades only in favorable regimes, HTF EMA prevents counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA trend filter (ER=10, FAST=2, SLOW=30)
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [0] * n
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # RSI(14) momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    # Multi-timeframe: 1w EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_market = chop[i] < 61.8
        # Avoid overextended RSI: long only if RSI < 70, short only if RSI > 30
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        # HTF trend filter: price above/below 1w EMA(34)
        htf_uptrend = close[i] > ema_34_1w_aligned[i]
        htf_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA
            if close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above KAMA
            if close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry: price crosses KAMA with RSI and regime confirmation
            bullish_entry = (close[i] > kama[i]) and rsi_not_overbought and trending_market and htf_uptrend
            bearish_entry = (close[i] < kama[i]) and rsi_not_oversold and trending_market and htf_downtrend
            
            if bullish_entry:
                position = 1
                signals[i] = 0.25
            elif bearish_entry:
                position = -1
                signals[i] = -0.25
    
    return signals