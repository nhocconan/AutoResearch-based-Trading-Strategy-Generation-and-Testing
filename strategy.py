#!/usr/bin/env python3
"""
Hypothesis: 1h primary with 4h HMA trend filter + RSI pullback entries + ATR stoploss
- 4h HMA(21) determines overall trend direction (HTF filter)
- 1h RSI(14) enters on pullbacks (RSI 35-65 range for trend continuation)
- ATR(14) stoploss at 2*ATR from entry
- Z-score(20) filter to avoid extreme entries
- Discrete position sizing: 0.0, ±0.25, ±0.35
- MUST generate trades on all 3 symbols (loose enough conditions)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_zscore_1h_v3"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2 * wma1 - wma2
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (CRITICAL RULE #1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)  # auto shift(1)
    
    # 1h indicators - compute BEFORE loop (CRITICAL RULE #8)
    close_s = pd.Series(close)
    
    # HMA(21) on 1h for local trend
    hma_1h = calculate_hma(close, 21)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3 = np.abs(low - np.roll(close, 1))
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Z-score(20) for extreme filter
    sma_20 = close_s.rolling(20, min_periods=20).mean().values
    std_20 = close_s.rolling(20, min_periods=20).std().values
    zscore = (close - sma_20) / (std_20 + 1e-10)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend from 4h HMA (aligned properly - no look-ahead)
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Entry conditions - LOOSE enough to generate trades (CRITICAL RULE #9)
        if hma_trend > 0:  # Long bias from HTF
            # Enter on pullback (RSI not overbought) + not at extreme high
            if rsi[i] < 65 and zscore[i] < 2.0:
                if position_side <= 0:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
            elif position_side == 1:
                # Hold position
                signals[i] = SIZE
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                
                # Stoploss: 2*ATR against entry
                if close[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position_side = 0
                
                # Trail stop at 1R profit
                elif close[i] > entry_price + 1.0 * atr[i]:
                    trail_stop = highest_since_entry - 1.0 * atr[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                
                # Take profit at 2R: reduce to half
                elif close[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = HALF_SIZE
        
        elif hma_trend < 0:  # Short bias from HTF
            # Enter on pullback (RSI not oversold) + not at extreme low
            if rsi[i] > 35 and zscore[i] > -2.0:
                if position_side >= 0:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
            elif position_side == -1:
                # Hold position
                signals[i] = -SIZE
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                
                # Stoploss: 2*ATR against entry
                if close[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position_side = 0
                
                # Trail stop at 1R profit
                elif close[i] < entry_price - 1.0 * atr[i]:
                    trail_stop = lowest_since_entry + 1.0 * atr[i]
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                
                # Take profit at 2R: reduce to half
                elif close[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = -HALF_SIZE
        
        # Trend reversal: exit position
        else:
            if position_side != 0:
                signals[i] = 0.0
                position_side = 0
    
    return signals