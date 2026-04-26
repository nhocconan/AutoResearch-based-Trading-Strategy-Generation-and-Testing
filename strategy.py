#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filtered_With_Volume_And_Chop_Regime_v2
Hypothesis: Daily KAMA trend with volume confirmation and choppiness regime filter.
KAMA adapts to market noise, reducing whipsaw in choppy regimes. Volume confirms institutional participation.
Choppiness filter avoids trend-following in ranging markets. Designed for BTC/ETH in both bull/bear markets.
Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA(10) on 1d
    close_1d = close  # prices are already 1d
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = kama  # already 1d
    
    # Calculate 1w EMA(10) for trend filter
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Choppiness regime: CHOP(14) > 61.8 = ranging (avoid trend signals)
    # True range
    tr_chop = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_chop[0] = high[0] - low[0]  # first bar
    atr_chop = pd.Series(tr_chop).rolling(window=14, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_chop.sum() / (hh - ll)) / np.log10(14), 50)
    chop_sum = pd.Series(chop).rolling(window=14, min_periods=14).sum().values
    chop = chop_sum  # already summed over window
    chop_regime = chop > 61.8  # True = ranging, avoid trend signals
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA seed, ATR, volume MA, chop
    start_idx = max(10, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or
            np.isnan(ema_10_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        ema_1w_val = ema_10_1w_aligned[i]
        vol_spike = volume_spike[i]
        is_chopping = chop_regime[i]  # True if ranging
        
        if position == 0:
            # Long: price > KAMA AND 1w trend up AND volume spike AND NOT chopping
            long_signal = (close_val > kama_val) and (close_val > ema_1w_val) and vol_spike and (not is_chopping)
            
            # Short: price < KAMA AND 1w trend down AND volume spike AND NOT chopping
            short_signal = (close_val < kama_val) and (close_val < ema_1w_val) and vol_spike and (not is_chopping)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < KAMA OR trend flips down OR ATR stoploss
            if (close_val < kama_val) or (close_val < ema_1w_val) or (close_val < entry_price - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > KAMA OR trend flips up OR ATR stoploss
            if (close_val > kama_val) or (close_val > ema_1w_val) or (close_val > entry_price + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filtered_With_Volume_And_Chop_Regime_v2"
timeframe = "1d"
leverage = 1.0