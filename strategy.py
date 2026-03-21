#!/usr/bin/env python3
"""
EXPERIMENT #025 - KAMA Adaptive Trend + RSI Pullback + BB Regime Filter
====================================================================================
Hypothesis: Replace HMA with KAMA for adaptive trend detection that adjusts smoothing
based on market efficiency (less whipsaw in chop, faster in trends). Use 1h entries
instead of 15m for cleaner signals. Add Bollinger Band width regime filter to avoid
trading in low-volatility chop where strategies bleed. Tighter 1.5*ATR stoploss for
quicker exits on failed trades.

Key improvements over #021:
- KAMA(10,2,30) instead of HMA - adapts to market efficiency ratio
- 1h entries instead of 15m - fewer false signals, less noise
- BB Width regime filter - skip trades when BB width < 2% (choppy market)
- Tighter stoploss: 1.5*ATR instead of 2.0*ATR - preserve capital faster
- Fixed discrete sizing: 0.25/0.35 - less churn than dynamic ATR sizing
- 4h trend via 16-bar KAMA on 1h data (no resampling look-ahead)

Why this might beat Sharpe=11.523:
- KAMA reduces whipsaws in ranging markets better than HMA
- 1h timeframe has cleaner trends than 15m with fewer false breakouts
- BB regime filter avoids low-volatility bleed periods
- Tighter stops preserve more capital on reversals
"""

import numpy as np
import pandas as pd

name = "mtf_kama_rsi_bb_regime_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA with first close price after warmup
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and width"""
    n = len(close)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_dev * std
    lower = mean - std_dev * std
    
    # BB Width = (Upper - Lower) / Middle
    bb_width = np.zeros(n)
    mask = mean > 0
    bb_width[mask] = (upper[mask] - lower[mask]) / mean[mask]
    
    # BB Percentile = (Close - Lower) / (Upper - Lower)
    bb_pct = np.zeros(n)
    mask2 = (upper - lower) > 0
    bb_pct[mask2] = (close[mask2] - lower[mask2]) / (upper[mask2] - lower[mask2])
    
    return upper, lower, bb_width, bb_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # KAMA for trend - use 16 bars for 4h equivalent on 1h data
    kama_fast_1h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow_1h = calculate_kama(close, period=30, fast_period=2, slow_period=30)
    
    # Bollinger Bands for regime filter
    bb_upper, bb_lower, bb_width, bb_pct = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 65    # Exit long when overbought
    RSI_EXIT_SHORT = 35   # Exit short when oversold
    
    # BB Width regime filter thresholds
    BB_WIDTH_LOW = 0.02   # Below this = choppy market, don't trade
    BB_WIDTH_HIGH = 0.15  # Above this = high volatility, be cautious
    
    # ATR stoploss multiplier (tighter than #021)
    ATR_STOP_MULT = 1.5   # Tighter stoploss
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(80, 30, 14, 20)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        # 4h-equivalent trend on 1h data (16 bars = 4h)
        if kama_fast_1h[i] > kama_slow_1h[i] and close[i] > kama_fast_1h[i]:
            trend = 1  # Bullish
        elif kama_fast_1h[i] < kama_slow_1h[i] and close[i] < kama_fast_1h[i]:
            trend = -1  # Bearish
        else:
            trend = 0  # No clear trend
        
        rsi_val = rsi_1h[i]
        bb_width_val = bb_width[i]
        bb_pct_val = bb_pct[i]
        atr = atr_1h[i]
        price = close[i]
        
        # BB Width regime filter - avoid choppy markets
        if bb_width_val < BB_WIDTH_LOW:
            # Choppy market - exit existing positions
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Stoploss check
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Stoploss check
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # New entry logic
        if trend == 1:  # Uptrend
            # RSI pullback entry in uptrend
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # Downtrend
            # RSI rally entry in downtrend
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals