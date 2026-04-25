#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index(14) for regime filtering.
Enter long when price > KAMA, RSI > 50, and CHOP > 61.8 (ranging market -> mean reversion long).
Enter short when price < KAMA, RSI < 50, and CHOP > 61.8 (ranging market -> mean reversion short).
Exit when opposite condition occurs. Uses discrete sizing (0.25) to limit fee churn.
Designed for 1d timeframe with ~10-25 trades/year, works in bull/bear by following KAMA trend
and fading extremes in ranging markets identified by high Choppiness.
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
    
    # 1w data for HTF trend filter (optional stronger filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA20 for stronger trend filter (only trade in alignment with weekly trend)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA (10, 2, 30) - ER = 10, fast=2, slow=30
    close_s = pd.Series(close)
    direction = abs(close_s - close_s.shift(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = pd.DataFrame({'high': high, 'low': low, 'close': close})
    atr['tr0'] = atr['high'] - atr['low']
    atr['tr1'] = abs(atr['high'] - atr['close'].shift())
    atr['tr2'] = abs(atr['low'] - atr['close'].shift())
    atr['tr'] = atr[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr_sum = atr['tr'].rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Volume confirmation: current volume > 1.3x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.3 * vol_ma.values)
    
    # Align HTF indicators
    kama_aligned = align_htf_to_ltf(prices, prices, kama)
    rsi_aligned = align_htf_to_ltf(prices, prices, rsi.values)
    chop_aligned = align_htf_to_ltf(prices, prices, chop.values)
    volume_spike_aligned = align_htf_to_ltf(prices, prices, volume_spike.values)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 30 for KAMA stability, 14 for RSI/CHOP, 20 for volume
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, in ranging market, volume spike, weekly uptrend
            long_setup = (close[i] > kama_aligned[i] and 
                         rsi_aligned[i] > 50 and 
                         in_range and 
                         volume_spike_aligned[i] and
                         close[i] > ema_20_1w_aligned[i])
            # Short: price < KAMA, RSI < 50, in ranging market, volume spike, weekly downtrend
            short_setup = (close[i] < kama_aligned[i] and 
                          rsi_aligned[i] < 50 and 
                          in_range and 
                          volume_spike_aligned[i] and
                          close[i] < ema_20_1w_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: opposite setup OR trend breaks (price < KAMA) OR chop drops (trending)
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] < 40 or 
                chop_aligned[i] < 50 or
                close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: opposite setup OR trend breaks (price > KAMA) OR chop drops (trending)
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] > 60 or 
                chop_aligned[i] < 50 or
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0