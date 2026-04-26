#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_MeanReversion_v1
Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI mean-reversion signals. In uptrend (price > KAMA), look for RSI < 30 for long entries.
In downtrend (price < KAMA), look for RSI > 70 for short entries. Volume spike confirms institutional
participation. This combines trend-following with mean-reversion pullbacks, working in both bull
and bear markets by adapting to the prevailing trend via KAMA. Designed for 50-150 total trades
over 4 years with discrete sizing (0.25) to minimize fee drag.
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
    
    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d KAMA (ER=10, fast=2, slow=30) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    change = abs(close_1d.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = close_1d.copy()
    for i in range(1, len(kama)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama.iloc[i-1])
    kama_1d = kama.values
    
    # Calculate 12h RSI(14) for mean-reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA calculation (10), RSI (14), volume MA (20)
    start_idx = max(10, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_1d_aligned[i]
        close_val = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price > KAMA (uptrend) or < KAMA (downtrend)
        uptrend = close_val > kama_val
        downtrend = close_val < kama_val
        
        if position == 0:
            # Long: RSI < 30 (oversold) in uptrend with volume spike
            long_signal = (rsi_val < 30) and uptrend and vol_spike
            
            # Short: RSI > 70 (overbought) in downtrend with volume spike
            short_signal = (rsi_val > 70) and downtrend and vol_spike
            
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
            # Exit: trend reversal or RSI > 70 (overbought)
            if close_val < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or RSI < 30 (oversold)
            if close_val > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_RSI_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0