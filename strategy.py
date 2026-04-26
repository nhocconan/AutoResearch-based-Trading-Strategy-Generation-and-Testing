#!/usr/bin/env python3
"""
1d_KAMA_Trend_Choppiness_Filter_v2
Hypothesis: Daily KAMA trend with choppiness regime filter and volume confirmation.
- Long when KAMA trending up, price > KAMA, chop > 61.8 (trending regime), volume > 1.3 * volume_ma(20)
- Short when KAMA trending down, price < KAMA, chop > 61.8 (trending regime), volume > 1.3 * volume_ma(20)
- Uses weekly trend filter (price > weekly EMA34 for longs, < for shorts) to avoid counter-trend whipsaws
- Chop filter ensures we only trade in trending markets (avoid ranging conditions)
- Volume confirmation reduces false signals
- Designed for low frequency (target 7-25 trades/year) to minimize fee drag on 1d timeframe
- Exit on opposite KAMA cross or trend/chop regime change
- Novelty: Combines adaptive trend (KAMA) with regime filter (choppiness) and HTF trend for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter (needs completed weekly candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Weekly trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    weekly_trend = np.where(ema_34_1w_aligned > 0, 
                            np.where(close > ema_34_1w_aligned, 1, -1), 
                            0)
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio: ER = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    net_change = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(net_change)):
        net_change[i] = np.abs(close[i] - close[i-10]) if i >= 10 else np.abs(close[i] - close[0])
    er = np.where(volatility > 0, net_change / volatility, 0)
    # Smoothing constants: sc = [ER * (fastest - slowest) + slowest]^2
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate Choppiness Index (CHOP)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Max/min close over period
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(TR14) / (maxClose - minClose)) / log10(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_close = max_close - min_close
    chop = np.where(range_close > 0, 100 * np.log10(sum_tr / range_close) / np.log10(14), 50)
    # Regime filter: chop > 61.8 = trending (we want to trade), chop < 38.2 = ranging
    chop_regime = chop > 61.8
    
    # Volume filter: volume > 1.3 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for weekly EMA, 14 for chop/ATR, 20 for volume MA)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA trend direction
        kama_trend = 0
        if i >= 1:
            if kama[i] > kama[i-1]:
                kama_trend = 1  # KAMA rising
            elif kama[i] < kama[i-1]:
                kama_trend = -1  # KAMA falling
        
        # Price vs KAMA
        price_vs_kama = 0
        if close[i] > kama[i]:
            price_vs_kama = 1  # Price above KAMA
        elif close[i] < kama[i]:
            price_vs_kama = -1  # Price below KAMA
        
        # Entry conditions
        if position == 0:
            # Long: KAMA trending up, price > KAMA, chop > 61.8 (trending), weekly uptrend, volume spike
            if (kama_trend == 1 and price_vs_kama == 1 and chop_regime[i] and 
                weekly_trend[i] == 1 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down, price < KAMA, chop > 61.8 (trending), weekly downtrend, volume spike
            elif (kama_trend == -1 and price_vs_kama == -1 and chop_regime[i] and 
                  weekly_trend[i] == -1 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA trend turns down OR price < KAMA OR chop < 61.8 (ranging) OR weekly trend turns down
            if (kama_trend == -1 or price_vs_kama == -1 or not chop_regime[i] or 
                weekly_trend[i] == -1):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA trend turns up OR price > KAMA OR chop < 61.8 (ranging) OR weekly trend turns up
            if (kama_trend == 1 or price_vs_kama == 1 or not chop_regime[i] or 
                weekly_trend[i] == 1):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Choppiness_Filter_v2"
timeframe = "1d"
leverage = 1.0