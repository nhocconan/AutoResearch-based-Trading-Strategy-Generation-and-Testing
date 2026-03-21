#!/usr/bin/env python3
"""
Hypothesis: 1h primary with 4h HMA trend + RSI pullback + Z-score regime filter.
Adapted from proven mtf_hma_rsi_zscore_v1 (Sharpe=5.4) but for 1h timeframe.
4h HMA(21) defines macro trend, 1h RSI(14) finds pullback entries, Z-score(20)
filters extreme moves. ATR(14) stoploss at 2.5*ATR protects during crashes.
SIZE=0.28 discrete levels balance trade frequency with fee costs.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_zscore_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # 1h indicators - all computed before loop (Rule 8)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    
    # ATR(14)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # HMA(16) for local trend
    hma16 = calculate_hma(close, 16)
    
    # Z-score(20) for regime detection
    sma20 = close_s.rolling(20, min_periods=20).mean().values
    std20 = close_s.rolling(20, min_periods=20).std().values
    zscore = (close - sma20) / np.where(std20 > 0, std20, 1.0)
    
    # EMA(50) for additional trend confirmation
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.28
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend: 4h HMA slope and price position
        htf_bullish = hma_4h_aligned[i] > hma_4h_aligned[i-1] and close[i] > hma_4h_aligned[i]
        htf_bearish = hma_4h_aligned[i] < hma_4h_aligned[i-1] and close[i] < hma_4h_aligned[i]
        
        # Local trend: HMA16 vs EMA50
        local_bullish = hma16[i] > ema50[i]
        local_bearish = hma16[i] < ema50[i]
        
        # Z-score regime: avoid extreme extensions
        zscore_normal = abs(zscore[i]) < 1.5  # not overextended
        zscore_oversold = zscore[i] < -1.0  # potential long entry
        zscore_overbought = zscore[i] > 1.0  # potential short entry
        
        # RSI pullback entries (not extreme)
        rsi_pullback_long = 40 < rsi[i] < 55  # pullback in uptrend
        rsi_pullback_short = 45 < rsi[i] < 60  # pullback in downtrend
        rsi_oversold = rsi[i] < 35  # deep oversold
        rsi_overbought = rsi[i] > 65  # deep overbought
        
        # Stoploss and trailing logic (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2.5 * atr[i]
            initial_stop = entry_price - 2.5 * atr[i]
            if close[i] < max(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            initial_stop = entry_price + 2.5 * atr[i]
            if close[i] > min(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long: HTF bullish + local bullish + pullback or oversold
            if htf_bullish and local_bullish and zscore_normal:
                if rsi_pullback_long or rsi_oversold:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
            
            # Short: HTF bearish + local bearish + pullback or overbought
            elif htf_bearish and local_bearish and zscore_normal:
                if rsi_pullback_short or rsi_overbought:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
        else:
            # Hold position - maintain current signal
            signals[i] = signals[i-1]
    
    return signals