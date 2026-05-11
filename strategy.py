#!/usr/bin/env python3
"""
4h_Liquidity_Grab_Reversal_v1
Hypothesis: Uses liquidity grab (equal highs/lows) as reversal signals in choppy markets.
In high choppiness (Choppiness > 61.8), we look for liquidity sweeps:
- Bearish liquidity grab: new high above recent high followed by close below prior high
- Bullish liquidity grab: new low below recent low followed by close above prior low
Entries occur on the next bar after confirmation. Exits on RSI mean reversion.
Designed for low trade frequency by requiring both liquidity grab and high choppiness.
Works in both bull and bear markets as liquidity grabs often precede reversals.
"""

name = "4h_Liquidity_Grab_Reversal_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- RSI (14-period) on 4h close ---
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # --- Choppiness Index (14-period) on 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    
    # Align Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # --- Liquidity Grab Detection ---
    # Lookback period for recent highs/lows
    lookback = 20
    
    # Arrays to track liquidity grab signals
    bullish_grab = np.zeros(n, dtype=bool)
    bearish_grab = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Recent swing high/low
        recent_high = np.max(high[i-lookback:i])
        recent_low = np.min(low[i-lookback:i])
        
        # Bullish liquidity grab: new low followed by close above recent low
        if low[i] < recent_low and close[i] > recent_low:
            bullish_grab[i] = True
            
        # Bearish liquidity grab: new high followed by close below recent high
        if high[i] > recent_high and close[i] < recent_high:
            bearish_grab[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(chop_aligned[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy/ranging markets
        choppy_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for liquidity grab signals in choppy markets
            if choppy_market:
                if bullish_grab[i]:
                    signals[i] = 0.25
                    position = 1
                elif bearish_grab[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions: RSI mean reversion or regime change
            if position == 1:
                # Exit long: RSI > 50 (overbought) or market becomes trending
                trending_market = chop_aligned[i] < 38.2
                exit_signal = (rsi[i] > 50) or trending_market
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 50 (oversold) or market becomes trending
                trending_market = chop_aligned[i] < 38.2
                exit_signal = (rsi[i] < 50) or trending_market
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals