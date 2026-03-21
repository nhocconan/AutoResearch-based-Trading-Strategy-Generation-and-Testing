#!/usr/bin/env python3
"""
EXPERIMENT #040 - MTF HMA+MACD+RSI+ADX+ATR Dynamic Sizing (15m+4h Clean v2)
==================================================================================================
Hypothesis: Current #040 has MTF alignment bugs (manual resampling instead of mtf_data helper).
This version fixes MTF alignment using proper mtf_data helper, simplifies indicator stack,
and uses ATR-based dynamic position sizing for better risk control.

Key changes from current #040:
- MTF: Use mtf_data helper (get_htf_data + align_htf_to_ltf) for proper 4h alignment
- Simpler indicator stack: HMA + MACD + RSI + ADX (remove KAMA, Z-score, BBW, Supertrend)
- Entry: MACD histogram cross + RSI pullback in 4h HMA trend direction
- Position sizing: ATR-based dynamic (base * target_vol / current_vol), capped at 0.35
- Stoploss: 2.5*ATR (slightly looser to avoid noise exits)
- Take profit: 2R with trail at 1R

Why this should work:
- Proper MTF alignment eliminates lookahead bias bugs
- MACD histogram provides cleaner momentum signals than RSI alone
- ATR dynamic sizing reduces position in high volatility (drawdown control)
- Based on proven 15m timeframe from #031, #034, #035
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_macd_rsi_adx_atr_dynamic_15m_4h_v2"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half, min_periods=half).apply(
        lambda x: np.dot(x, np.arange(1, half+1)) / np.sum(np.arange(1, half+1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(1, period+1)) / np.sum(np.arange(1, period+1)), raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).rolling(window=sqrt_p, min_periods=sqrt_p).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_p+1)) / np.sum(np.arange(1, sqrt_p+1)), raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return np.nan_to_num(macd_line, nan=0.0), np.nan_to_num(signal_line, nan=0.0), np.nan_to_num(histogram, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    # Wilder's smoothing
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=0.0)


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return np.nan_to_num(atr, nan=0.0)


def calculate_adx(high, low, close, period=14):
    """Calculate ADX"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return np.nan_to_num(adx, nan=0.0)


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h HTF data using mtf_data helper (MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h trend indicators
    hma_4h = calculate_hma(close_4h, period=21)
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
    
    # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 15m entry indicators
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    rsi_15m = calculate_rsi(close, period=14)
    atr_15m = calculate_atr(high, low, close, period=14)
    
    # Parameters
    BASE_SIZE = 0.30
    MAX_SIZE = 0.35
    MIN_SIZE = 0.15
    TARGET_ATR_PCT = 0.015  # Target 1.5% ATR for sizing
    ATR_STOP_MULT = 2.5
    TP_MULT = 2.0
    TRAIL_MULT = 1.0
    ADX_MIN = 20
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 55
    
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    extreme_price = np.zeros(n)
    
    first_valid = max(100, 40 * 4)  # Ensure 4h data is aligned
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        atr_pct = atr / price
        
        # Dynamic position sizing based on ATR
        vol_ratio = TARGET_ATR_PCT / max(atr_pct, 0.001)
        vol_ratio = np.clip(vol_ratio, 0.5, 1.5)
        position_size = np.clip(BASE_SIZE * vol_ratio, MIN_SIZE, MAX_SIZE)
        
        # 4h trend direction
        trend_4h = 0
        if hma_4h_aligned[i] > 0 and price > hma_4h_aligned[i]:
            trend_4h = 1
        elif hma_4h_aligned[i] > 0 and price < hma_4h_aligned[i]:
            trend_4h = -1
        
        adx_4h_val = adx_4h_aligned[i]
        
        # Manage existing positions
        if position_side[i-1] != 0:
            prev_side = position_side[i-1]
            prev_entry = entry_price[i-1] if entry_price[i-1] > 0 else price
            prev_tp = tp_triggered[i-1]
            prev_extreme = extreme_price[i-1] if extreme_price[i-1] > 0 else prev_entry
            
            # Update extreme price
            if prev_side == 1:
                current_extreme = max(prev_extreme, price)
            else:
                current_extreme = min(prev_extreme, price) if prev_extreme > 0 else price
            extreme_price[i] = current_extreme
            
            # Stoploss check
            if prev_side == 1:
                stop_price = prev_entry - ATR_STOP_MULT * atr
                if price < stop_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    extreme_price[i] = 0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = position_size * 0.5
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R after TP
                if prev_tp:
                    trail_price = current_extreme - TRAIL_MULT * ATR_STOP_MULT * atr
                    if price < trail_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        extreme_price[i] = 0
                        continue
            else:  # Short
                stop_price = prev_entry + ATR_STOP_MULT * atr
                if price > stop_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    extreme_price[i] = 0
                    continue
                
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -position_size * 0.5
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                if prev_tp:
                    trail_price = current_extreme + TRAIL_MULT * ATR_STOP_MULT * atr
                    if price > trail_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        extreme_price[i] = 0
                        continue
            
            # Hold position
            signals[i] = signals[i-1]
            position_side[i] = position_side[i-1]
            entry_price[i] = entry_price[i-1]
            tp_triggered[i] = tp_triggered[i-1]
            extreme_price[i] = extreme_price[i-1]
            continue
        
        # Entry logic: MACD + RSI pullback in 4h trend direction
        rsi = rsi_15m[i]
        hist = macd_hist[i]
        hist_prev = macd_hist[i-1] if i > 0 else 0
        
        # ADX filter - only trade when 4h trend is strong
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Long entry: 4h uptrend + MACD bullish + RSI pullback
        if trend_4h == 1:
            if hist > 0 and hist_prev <= 0:  # MACD cross above signal
                if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:  # RSI pullback zone
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    extreme_price[i] = price
                    continue
        
        # Short entry: 4h downtrend + MACD bearish + RSI pullback
        elif trend_4h == -1:
            if hist < 0 and hist_prev >= 0:  # MACD cross below signal
                if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:  # RSI pullback zone
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    extreme_price[i] = price
                    continue
        
        # No position
        signals[i] = 0.0
        position_side[i] = 0
    
    return signals