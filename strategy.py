#!/usr/bin/env python3
"""
EXPERIMENT #037 - MTF HMA+RSI+ZSCORE 15m+4h v1
==================================================================================================
Hypothesis: Move to 15m base timeframe with 4h trend filter for more trade opportunities.
The current best (30m+4h) has Sharpe=1.153. By moving to 15m base:
- More trade signals (15m has 2x the bars of 30m)
- Tighter stoploss possible (2.0*ATR vs 2.5*ATR) due to lower timeframe
- Add Z-score(20) filter to avoid chasing extreme moves
- Base position size 0.25 (more conservative than 0.30-0.35)
- Simpler exit logic (remove complex TP/trailing, just ATR stop)

Why this should beat current best:
- 15m timeframe balances signal quality vs trade count
- Z-score filter prevents entries at extremes (common failure mode)
- Tighter stops reduce drawdown per trade
- Lower base size (0.25) provides buffer during drawdowns
- Remove TP complexity that caused issues in #036
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_zscore_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_fast_15m = calculate_hma(close, period=21)
    hma_slow_15m = calculate_hma(close, period=48)
    zscore_15m = calculate_zscore(close, period=20)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_fast_4h = calculate_hma(c_4h, period=21)
        hma_slow_4h = calculate_hma(c_4h, period=48)
        rsi_4h = calculate_rsi(c_4h, period=14)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_fast_4h)
        hma_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_slow_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    except Exception:
        # Fallback if mtf_data fails
        hma_fast_4h_aligned = np.zeros(n)
        hma_slow_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n) + 50
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (conservative for 15m)
    BASE_SIZE = 0.25
    TARGET_ATR_PCT = 0.015  # Target 1.5% ATR as % of price for 15m
    
    # RSI thresholds for pullback entries (15m specific)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score filter (avoid entries at extremes)
    ZSCORE_MAX = 1.5  # Don't enter if price > 1.5 std from mean
    
    # ATR stoploss multiplier (tighter for 15m)
    ATR_STOP_MULT = 2.0
    
    # Minimum volatility filter
    MIN_ATR_PCT = 0.003
    
    first_valid = max(200, 48, 30)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    entry_atr = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Calculate ATR as % of price for dynamic sizing
        atr_pct = atr_15m[i] / close[i] if close[i] > 0 else 0
        
        # Skip if volatility is extremely low
        if atr_pct < MIN_ATR_PCT:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = 0
            continue
        
        # Dynamic position sizing based on current volatility
        if atr_pct > 0:
            vol_adjustment = min(TARGET_ATR_PCT / atr_pct, 1.5)  # Cap at 1.5x
            size_full = min(BASE_SIZE * vol_adjustment, 0.40)  # Max 40%
        else:
            size_full = BASE_SIZE
        
        # Get aligned MTF values
        hma_fast_4h_val = hma_fast_4h_aligned[i] if i < len(hma_fast_4h_aligned) else 0
        hma_slow_4h_val = hma_slow_4h_aligned[i] if i < len(hma_slow_4h_aligned) else 0
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        
        # 4h trend filter: HMA fast > HMA slow = bullish
        trend_4h = 0
        if hma_fast_4h_val > 0 and hma_slow_4h_val > 0:
            if hma_fast_4h_val > hma_slow_4h_val:
                trend_4h = 1
            elif hma_fast_4h_val < hma_slow_4h_val:
                trend_4h = -1
        
        # 4h RSI filter (avoid extreme overbought/oversold)
        rsi_4h_ok = True
        if trend_4h == 1 and rsi_4h_val > 75:
            rsi_4h_ok = False
        elif trend_4h == -1 and rsi_4h_val < 25:
            rsi_4h_ok = False
        
        # Check stoploss for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr_15m[i - 1]
            
            price = close[i]
            
            # Stoploss check (2.0*ATR from entry)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
            
            # Check if trend changed - exit position
            if trend_4h != prev_side and trend_4h != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                entry_atr[i] = 0
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            entry_atr[i] = entry_atr[i - 1]
            continue
        
        # Entry logic: 4h trend + 15m RSI pullback + Z-score filter
        price = close[i]
        zscore_val = zscore_15m[i]
        
        # Z-score filter: don't enter if price is too far from mean
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 15m HMA alignment check
        hma_aligned_15m = 0
        if hma_fast_15m[i] > hma_slow_15m[i]:
            hma_aligned_15m = 1
        elif hma_fast_15m[i] < hma_slow_15m[i]:
            hma_aligned_15m = -1
        
        if trend_4h == 1 and rsi_4h_ok and hma_aligned_15m == 1:  # Bullish trend
            # RSI pullback on 15m (not overbought)
            if RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX:
                signals[i] = size_full
                position_side[i] = 1
                entry_price[i] = price
                entry_atr[i] = atr_15m[i]
                
        elif trend_4h == -1 and rsi_4h_ok and hma_aligned_15m == -1:  # Bearish trend
            # RSI pullback on 15m (not oversold)
            if RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX:
                signals[i] = -size_full
                position_side[i] = -1
                entry_price[i] = price
                entry_atr[i] = atr_15m[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals