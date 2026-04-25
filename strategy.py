#!/usr/bin/env python3
"""
1d KAMA Direction + RSI + Chop Filter
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) adapts to market noise - 
efficient in trending markets, flat in ranging markets. Combined with RSI for 
momentum and Choppiness Index for regime filtering, this captures strong trends 
while avoiding whipsaws in choppy conditions. Works in bull markets (KAMA up) 
and bear markets (KAMA down) by using KAMA direction as primary filter.
Designed for BTC/ETH with 30-100 total trades over 4 years (7-25/year) to minimize 
fee drag while capturing major moves.
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
    
    # Get weekly data for HTF trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for weekly EMA
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for KAMA
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'])
    # KAMA parameters: ER=10, Fast=2, Slow=30
    change = abs(close_1d - close_1d.shift(10))
    volatility = abs(close_1d - close_1d.shift(1)).rolling(10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close_1d.iloc[0]]  # Initialize with first value
    for i in range(1, len(close_1d)):
        if pd.isna(sc.iloc[i]):
            kama.append(kama[-1])
        else:
            kama.append(kama[-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[-1]))
    kama_1d = np.array(kama)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d timeframe
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Calculate Choppiness Index on 1d timeframe
    atr_1d = []
    tr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            tr = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        tr_1d.append(tr)
        atr_1d.append(np.mean(tr_1d[-14:]) if len(tr_1d) >= 14 else np.nan)
    
    atr_1d = np.array(atr_1d)
    max_high_1d = pd.Series(df_1d['high']).rolling(14, min_periods=14).max().values
    min_low_1d = pd.Series(df_1d['low']).rolling(14, min_periods=14).min().values
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if atr_1d[i] > 0 and (max_high_1d[i] - min_low_1d[i]) > 0:
            chop_1d[i] = 100 * np.log10(sum(atr_1d[i-13:i+1]) / np.log(14) / (max_high_1d[i] - min_low_1d[i]))
        else:
            chop_1d[i] = 50.0  # Neutral when undefined
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = 50  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # Regime filter: Choppiness Index < 61.8 = trending (avoid choppy markets)
        trending_regime = chop_val < 61.8
        
        # KAMA direction: price above/below KAMA indicates trend direction
        kama_uptrend = curr_close > kama_val
        kama_downtrend = curr_close < kama_val
        
        # RSI momentum: avoid extreme overbought/oversold for better entries
        rsi_momentum = (rsi_val > 30) and (rsi_val < 70)
        
        # Weekly trend alignment: ensure alignment with higher timeframe
        weekly_uptrend = curr_close > ema_50_val
        weekly_downtrend = curr_close < ema_50_val
        
        if position == 0:
            # Look for entry signals
            # Long: price above KAMA, RSI not oversold, weekly uptrend, trending regime
            long_entry = (kama_uptrend and rsi_momentum and weekly_uptrend and trending_regime)
            # Short: price below KAMA, RSI not overbought, weekly downtrend, trending regime
            short_entry = (kama_downtrend and rsi_momentum and weekly_downtrend and trending_regime)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price crosses below KAMA OR weekly trend turns down
            if curr_close < kama_val or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR weekly trend turns up
            if curr_close > kama_val or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0