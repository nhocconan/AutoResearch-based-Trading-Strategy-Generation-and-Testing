#!/usr/bin/env python3
"""
EXPERIMENT #014 - 1h Bollinger Squeeze Breakout with 4h HMA Trend Filter
=========================================================================
Hypothesis: Bollinger Band squeeze on 1h identifies low-volatility consolidation
periods that precede explosive moves. Combined with 4h HMA trend filter, we only
take breakouts in the direction of the higher timeframe trend. RSI filter avoids
entering at overbought/oversold extremes.

Key improvements:
- 4h HMA trend filter (faster than EMA, smoother than SMA)
- 1h Bollinger squeeze detection (BB width < 20th percentile of last 100 bars)
- RSI(14) filter: avoid long if RSI>70, avoid short if RSI<30
- ATR-based stoploss: exit when price moves 2.5*ATR against position
- Take profit: reduce to half position at 2R profit
- Discrete signal levels (0.0, ±0.25, ±0.35) to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_bb_squeeze_hma_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, adjust=False).mean()
    hma = (2 * wma_half - wma_full).ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    return upper.values, lower.values, width.values


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD 4h HTF DATA ONCE BEFORE LOOP ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_prev = calculate_hma(close_4h, 21)
    
    # Align 4h HMA to 1h timeframe (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_4h_prev_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_prev)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    # Bollinger Bands for squeeze detection
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # BB width percentile for squeeze detection
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x.dropna(), 20) if len(x.dropna()) >= 10 else np.nan
    ).values
    
    # RSI for entry filter
    rsi = calculate_rsi(close, period=14)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, period=14)
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    
    # Position tracking for stoploss/takeprofit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_atr = 0.0
    
    # Position sizing
    SIZE_ENTRY = 0.35  # 35% for new entries
    SIZE_HALF = 0.175  # 17.5% for take profit (half position)
    
    # Minimum bars for valid signals
    min_bars = max(100, len(df_4h) * 16)  # Ensure we have enough 4h data aligned
    
    for i in range(min_bars, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_width[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === 4h TREND FILTER ===
        # HMA slope direction
        hma_slope = hma_4h_aligned[i] - hma_4h_prev_aligned[i]
        trend_long = hma_slope > 0
        trend_short = hma_slope < 0
        
        # === 1h BOLLINGER SQUEEZE DETECTION ===
        # Squeeze: BB width below 20th percentile of last 100 bars
        is_squeeze = bb_width[i] < bb_width_percentile[i] if not np.isnan(bb_width_percentile[i]) else False
        
        # Breakout detection
        breakout_long = close[i] > bb_upper[i] and is_squeeze
        breakout_short = close[i] < bb_lower[i] and is_squeeze
        
        # === RSI FILTER ===
        rsi_ok_long = rsi[i] < 70  # Not overbought
        rsi_ok_short = rsi[i] > 30  # Not oversold
        
        # === STOPLOSS LOGIC ===
        if position_side == 1 and entry_atr > 0:
            # Long stoploss: price drops 2.5*ATR from entry
            stoploss_price = entry_price - 2.5 * entry_atr
            # Take profit: reduce to half at 2R (2 * 2.5*ATR = 5*ATR)
            takeprofit_price = entry_price + 5.0 * entry_atr
            
            if close[i] < stoploss_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue
            elif close[i] > takeprofit_price and signals[i-1] == SIZE_ENTRY:
                # Take profit: reduce to half position
                signals[i] = SIZE_HALF
                continue
        
        elif position_side == -1 and entry_atr > 0:
            # Short stoploss: price rises 2.5*ATR from entry
            stoploss_price = entry_price + 2.5 * entry_atr
            # Take profit: reduce to half at 2R
            takeprofit_price = entry_price - 5.0 * entry_atr
            
            if close[i] > stoploss_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue
            elif close[i] < takeprofit_price and signals[i-1] == -SIZE_ENTRY:
                # Take profit: reduce to half position
                signals[i] = -SIZE_HALF
                continue
        
        # === ENTRY SIGNALS ===
        if position_side == 0:
            # Long entry: 4h uptrend + 1h squeeze breakout + RSI filter
            if trend_long and breakout_long and rsi_ok_long:
                signals[i] = SIZE_ENTRY
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
            
            # Short entry: 4h downtrend + 1h squeeze breakout + RSI filter
            elif trend_short and breakout_short and rsi_ok_short:
                signals[i] = -SIZE_ENTRY
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
            else:
                signals[i] = 0.0
        
        # If already in position, maintain signal (unless TP/SL triggered above)
        elif position_side == 1:
            if signals[i] != SIZE_HALF:  # Not already reduced for TP
                signals[i] = SIZE_ENTRY
        elif position_side == -1:
            if signals[i] != -SIZE_HALF:  # Not already reduced for TP
                signals[i] = -SIZE_ENTRY
    
    return signals