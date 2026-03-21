#!/usr/bin/env python3
"""
EXPERIMENT #033 - MTF HMA+RSI Simplified 30m+4h v3
==================================================================================================
Hypothesis: Return to the winning formula from #022 (30m+4h HMA+RSI) but with:
- Cleaner signal logic (fewer condition changes = less churn)
- Better stoploss implementation (ATR-based, properly tracked)
- Discrete signal levels (0, ±0.25, ±0.35) to minimize fees
- 4h HMA trend filter + 30m RSI pullback entries
- Position sizing capped at 0.35 to control drawdown

Why this should work:
- #022 proved 30m+4h combination works (Sharpe=1.153)
- Simpler logic reduces signal churn and fees
- Proper ATR stops protect capital
- Discrete levels minimize unnecessary position changes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_30m_4h_simplified_v3"
timeframe = "30m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Base timeframe (30m) indicators
    atr_30m = calculate_atr(high, low, close, period=14)
    hma_30m = calculate_hma(close, period=21)
    rsi_30m = calculate_rsi(close, period=14)
    
    # Get 4h data using mtf_data helper
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend filter
        hma_4h = calculate_hma(c_4h, period=21)
        
        # Align 4h indicators to 30m timeframe
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
    except Exception:
        hma_4h_aligned = np.zeros(n)
        close_4h_aligned = np.zeros(n)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45
    RSI_SHORT_ENTRY = 55
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum bars for warmup
    first_valid = max(100, 21 * 2, 14 + 1)
    
    # Track position state
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get 4h trend direction
        hma_4h_val = hma_4h_aligned[i] if i < len(hma_4h_aligned) else 0
        close_4h_val = close_4h_aligned[i] if i < len(close_4h_aligned) else close[i]
        
        # 4h trend filter: price vs HMA
        trend_4h = 0
        if close_4h_val > hma_4h_val:
            trend_4h = 1
        elif close_4h_val < hma_4h_val:
            trend_4h = -1
        
        # 30m trend filter: price vs HMA
        trend_30m = 0
        if close[i] > hma_30m[i]:
            trend_30m = 1
        elif close[i] < hma_30m[i]:
            trend_30m = -1
        
        # Check existing position for stoploss
        if position_side != 0:
            if position_side == 1:
                # Long position
                if close[i] <= stoploss_price:
                    # Stoploss hit - close position
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    stoploss_price = 0.0
                    continue
            elif position_side == -1:
                # Short position
                if close[i] >= stoploss_price:
                    # Stoploss hit - close position
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    stoploss_price = 0.0
                    continue
            
            # Hold existing position
            signals[i] = position_side * SIZE_FULL
            continue
        
        # Entry logic
        # Long: 4h uptrend + 30m uptrend + RSI pullback
        if trend_4h == 1 and trend_30m == 1 and rsi_30m[i] <= RSI_LONG_ENTRY:
            signals[i] = SIZE_FULL
            position_side = 1
            entry_price = close[i]
            stoploss_price = entry_price - ATR_STOP_MULT * atr_30m[i]
        
        # Short: 4h downtrend + 30m downtrend + RSI pullback
        elif trend_4h == -1 and trend_30m == -1 and rsi_30m[i] >= RSI_SHORT_ENTRY:
            signals[i] = -SIZE_FULL
            position_side = -1
            entry_price = close[i]
            stoploss_price = entry_price + ATR_STOP_MULT * atr_30m[i]
        
        else:
            signals[i] = 0.0
    
    return signals