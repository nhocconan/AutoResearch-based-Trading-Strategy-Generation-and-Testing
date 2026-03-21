#!/usr/bin/env python3
"""
EXPERIMENT #004 - MTF HMA+RSI+ZSCORE (30m+4h v2)
==================================================================================================
Hypothesis: Simplify the proven mtf_hma_rsi_zscore_v1 (Sharpe=5.4) by:
- Using 30m as primary timeframe (between 15m and 1h, less noise than 15m)
- 4h HMA for trend direction (proven effective)
- 30m RSI for pullback entries (buy dips in uptrend)
- Z-score filter to avoid extreme deviations
- Cleaner position management with proper ATR stoploss

Why this should beat the failed #001-#003:
- HMA is more responsive than KAMA/DEMA for trend following
- RSI pullback entries work better than MACD/Stoch in trending markets
- Z-score avoids chasing extreme moves (reduces FOMO entries)
- 30m timeframe balances trade frequency vs noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_zscore_30m_4h_v2"
timeframe = "30m"
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
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
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
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    zscore_30m = calculate_zscore(close, period=20)
    hma_30m = calculate_hma(close, period=21)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(c_4h, period=21)
        rsi_4h = calculate_rsi(c_4h, period=14)
        
        # Align 4h indicators to 30m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    except Exception:
        hma_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n) + 50
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.0875
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45  # Buy when RSI pulls back to 45 in uptrend
    RSI_SHORT_ENTRY = 55  # Sell when RSI rallies to 55 in downtrend
    RSI_EXIT = 65  # Exit long when RSI > 65 (overbought)
    RSI_EXIT_SHORT = 35  # Exit short when RSI < 35 (oversold)
    
    # Z-score filter (avoid extreme deviations)
    ZSCORE_MAX = 2.0  # Don't enter if price is > 2 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 21, 14, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    stoploss_price = np.zeros(n)
    tp_level_reached = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # Get aligned 4h values
        hma_4h_val = hma_4h_aligned[i] if i < len(hma_4h_aligned) else close[i]
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0:
            if close[i] > hma_4h_val:
                trend_4h = 1
            elif close[i] < hma_4h_val:
                trend_4h = -1
        
        # Z-score filter
        zscore_val = zscore_30m[i]
        
        # Check existing positions first (stoploss, take profit, exit signals)
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_stop = stoploss_price[i - 1]
            prev_tp = tp_level_reached[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            current_price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, current_price)
                current_low = prev_low
            else:
                current_high = prev_high
                current_low = min(prev_low, current_price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check
            if prev_side == 1 and current_price < prev_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                tp_level_reached[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            if prev_side == -1 and current_price > prev_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                tp_level_reached[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Take profit: reduce to half at 2R
            if prev_side == 1 and not prev_tp:
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_30m[i]
                if current_price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    stoploss_price[i] = prev_entry + 1 * ATR_STOP_MULT * atr_30m[i]  # Trail to 1R
                    tp_level_reached[i] = 1
                    continue
            
            if prev_side == -1 and not prev_tp:
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_30m[i]
                if current_price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    stoploss_price[i] = prev_entry - 1 * ATR_STOP_MULT * atr_30m[i]  # Trail to 1R
                    tp_level_reached[i] = 1
                    continue
            
            # Trail stop after TP reached
            if prev_tp:
                if prev_side == 1:
                    trail_stop = current_high - ATR_STOP_MULT * atr_30m[i]
                    if current_price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        stoploss_price[i] = 0
                        tp_level_reached[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    else:
                        stoploss_price[i] = trail_stop
                
                if prev_side == -1:
                    trail_stop = current_low + ATR_STOP_MULT * atr_30m[i]
                    if current_price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        stoploss_price[i] = 0
                        tp_level_reached[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    else:
                        stoploss_price[i] = trail_stop
            
            # RSI exit signals
            if prev_side == 1 and rsi_30m[i] > RSI_EXIT:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                tp_level_reached[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            if prev_side == -1 and rsi_30m[i] < RSI_EXIT_SHORT:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                tp_level_reached[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Trend reversal exit
            if prev_side == 1 and trend_4h == -1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                tp_level_reached[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            if prev_side == -1 and trend_4h == 1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                tp_level_reached[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            stoploss_price[i] = stoploss_price[i - 1]
            tp_level_reached[i] = tp_level_reached[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # No position - check for entry
        current_price = close[i]
        
        # Long entry: 4h uptrend + RSI pullback + Z-score not extreme
        if trend_4h == 1 and rsi_30m[i] <= RSI_LONG_ENTRY and abs(zscore_val) < ZSCORE_MAX:
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = current_price
            stoploss_price[i] = current_price - ATR_STOP_MULT * atr_30m[i]
            tp_level_reached[i] = 0
            highest_since_entry[i] = current_price
            lowest_since_entry[i] = current_price
        
        # Short entry: 4h downtrend + RSI rally + Z-score not extreme
        elif trend_4h == -1 and rsi_30m[i] >= RSI_SHORT_ENTRY and abs(zscore_val) < ZSCORE_MAX:
            signals[i] = -SIZE_FULL
            position_side[i] = -1
            entry_price[i] = current_price
            stoploss_price[i] = current_price + ATR_STOP_MULT * atr_30m[i]
            tp_level_reached[i] = 0
            highest_since_entry[i] = current_price
            lowest_since_entry[i] = current_price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals