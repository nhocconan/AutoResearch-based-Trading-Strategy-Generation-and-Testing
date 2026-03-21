#!/usr/bin/env python3
"""
Hypothesis: 4h primary with 1d HTF trend filter reduces noise vs 15m/30m/1h failures.
Donchian(20) breakouts + RSI(14) pullback entries aligned with 1d HMA(21) trend.
ATR(14) stoploss at 2*ATR + trailing stop protects capital during drawdowns.
SIZE=0.25 discrete levels minimize fee churn while generating sufficient trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_rsi_4h_v1"
timeframe = "4h"
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
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    hma_1d = calculate_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # 4h indicators - all computed before loop (Rule 8)
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
    
    # Donchian(20)
    donchian_upper = pd.Series(high).rolling(20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # EMA(50) for additional trend filter
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend: price vs 1d HMA (Rule 2 - use aligned array)
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # Local trend: price vs EMA50
        local_bullish = close[i] > ema50[i]
        local_bearish = close[i] < ema50[i]
        
        # Donchian breakout signals (use previous bar to avoid look-ahead)
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # RSI filter - pullback entries (not extreme)
        rsi_ok_long = rsi[i] < 65  # not overbought
        rsi_ok_short = rsi[i] > 35  # not oversold
        
        # RSI extreme for counter-trend (optional entries)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Stoploss and trailing logic (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2 * atr[i]
            initial_stop = entry_price - 2 * atr[i]
            if close[i] < max(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2 * atr[i]
            initial_stop = entry_price + 2 * atr[i]
            if close[i] > min(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long: HTF bullish + (breakout OR pullback)
            if htf_bullish and local_bullish:
                if breakout_long and rsi_ok_long:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                elif rsi_oversold and close[i] > donchian_mid[i]:
                    # Pullback entry in uptrend
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
            
            # Short: HTF bearish + (breakout OR pullback)
            elif htf_bearish and local_bearish:
                if breakout_short and rsi_ok_short:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
                elif rsi_overbought and close[i] < donchian_mid[i]:
                    # Pullback entry in downtrend
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
        else:
            # Hold position
            signals[i] = signals[i-1]
    
    return signals