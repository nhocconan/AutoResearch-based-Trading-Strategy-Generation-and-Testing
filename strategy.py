#!/usr/bin/env python3
"""
EXPERIMENT #081 - SIMPLIFIED_TREND_PULLBACK_MTF_15M_4H_V1
==================================================================================================
Hypothesis: Recent ensemble failures (#070-#079) show complex voting creates excessive churn and fees.
This strategy uses a CLEAN trend-following approach with pullback entries, similar to #075's success.

Key design choices:
- 4h HMA(21) for primary trend direction (proven in #075 with Sharpe=0.277)
- 15m RSI(14) pullback entries in trend direction (buy dips in uptrend, sell rallies in downtrend)
- BBW(20) filter to avoid entering during extreme squeezes/expansions
- ATR(14) stoploss at 2.5*ATR with take-profit at 2R (reduce to half position)
- Position sizing: discrete levels (0.0, ±0.25, ±0.35) to minimize signal churn
- Volume confirmation: require 1.2x 20-bar average volume on entry

Why this should beat recent failures:
- Simpler logic = fewer signal flips = lower 0.10% round-trip fees
- Proper MTF alignment via mtf_data helper (46 strategies failed without this)
- Focus on ONE high-quality signal (trend + pullback) instead of voting chaos
- Based on #075's success pattern but cleaner implementation
- Conservative sizing (max 0.35) controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "simplified_trend_pullback_mtf_15m_4h_v1"
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
    
    hma = np.zeros(n)
    raw_vals = 2 * wma1 - wma2
    
    for i in range(sqrt_period - 1, n):
        window = raw_vals[i - sqrt_period + 1:i + 1]
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(window * weights) / np.sum(weights)
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI using Welles Wilder's method"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
    bbw = np.zeros(n)
    for i in range(period - 1, n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
    
    return upper, middle, lower, bbw


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.ones(n)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.ones(n)
    
    for i in range(period - 1, n):
        if vol_avg[i] > 0:
            vol_ratio[i] = volume[i] / vol_avg[i]
    
    return vol_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_ratio_15m = calculate_volume_ratio(volume, period=20)
    
    # 4h trend indicators using mtf_data helper (MANDATORY)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Generate signals with trend-following + pullback logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.18
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # BBW filter thresholds (avoid extreme volatility)
    BBW_MIN = 0.02
    BBW_MAX = 0.15
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Volume confirmation threshold
    VOL_MIN = 1.2
    
    first_valid = max(200, 14 * 2, 20, 28)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip if indicators are invalid
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get 4h trend direction
        hma_4h_val = hma_4h_aligned[i]
        bbw_4h_val = bbw_4h_aligned[i]
        
        # Determine 4h trend
        trend_4h = 0
        if hma_4h_val > 0:
            if close[i] > hma_4h_val:
                trend_4h = 1  # Bullish
            elif close[i] < hma_4h_val:
                trend_4h = -1  # Bearish
        
        # BBW filter - avoid extreme volatility regimes
        bbw_ok = BBW_MIN < bbw_15m[i] < BBW_MAX
        
        # Volume confirmation
        vol_confirmed = vol_ratio_15m[i] >= VOL_MIN
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, high[i])
                current_low = min(prev_low, low[i]) if prev_low > 0 else low[i]
            else:
                current_high = max(prev_high, high[i]) if prev_high > 0 else high[i]
                current_low = min(prev_low, low[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if low[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and high[i] >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if low[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if high[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and low[i] <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if high[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: trend-following with pullback
        rsi_val = rsi_15m[i]
        
        if trend_4h == 1 and bbw_ok and vol_confirmed:
            # Uptrend: buy pullback (RSI dips to 40-60 range)
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        elif trend_4h == -1 and bbw_ok and vol_confirmed:
            # Downtrend: sell rally (RSI rises to 40-60 range)
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        # No position
        if signals[i] == 0:
            position_side[i] = 0
    
    return signals