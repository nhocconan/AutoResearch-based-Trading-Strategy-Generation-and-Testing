#!/usr/bin/env python3
"""
EXPERIMENT #077 - Regime-Switching KAMA/RSI with 4h Trend Filter
==================================================================================================
Hypothesis: Complex voting ensembles failed due to churn and conflicting signals.
This version uses a cleaner regime-switching approach:
1. Bollinger Band Width percentile detects regime (low vol = trend, high vol = mean revert)
2. KAMA adapts to market efficiency (faster in trends, slower in chop)
3. RSI for mean reversion entries in high vol regime
4. 4h trend filter provides direction bias (proven in current best)
5. Discrete signal levels (0.0, ±0.25, ±0.35) minimize churn costs
6. Conservative position sizing (max 0.35) controls drawdown

Why this should beat current best (Sharpe=3.653):
- Regime detection avoids wrong strategy in wrong market
- KAMA is more adaptive than HMA/EMA
- 4h trend filter proven effective in current best
- Less churn than voting ensembles (#070, #071 failed with -60%+ DD)
- Based on #075 learnings (Sharpe=0.277) but cleaner logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_kama_rsi_mtf_15m_4h_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper_band = rolling_mean + std_mult * rolling_std
    lower_band = rolling_mean - std_mult * rolling_std
    band_width = (upper_band - lower_band) / rolling_mean
    
    # Handle division by zero
    band_width = np.nan_to_num(band_width, nan=0.0, posinf=0.0, neginf=0.0)
    
    return upper_band, lower_band, band_width


def calculate_bbw_percentile(band_width, lookback=100):
    """Calculate Bollinger Band Width percentile (regime indicator)"""
    n = len(band_width)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = band_width[i - lookback:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            percentile[i] = np.sum(valid_window <= band_width[i]) / len(valid_window)
        else:
            percentile[i] = 0.5
    
    return percentile


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
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 15m indicators for entry timing ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    kama_15m = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    _, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # ========== 4h indicators via mtf_data helper (CRITICAL) ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        kama_4h = calculate_kama(close_4h, period=10, fast_period=2, slow_period=30)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
    except Exception as e:
        # Fallback if mtf_data fails
        kama_4h_aligned = kama_15m
        st_4h_aligned = st_direction_15m
        bbw_4h_aligned = bbw_15m
    
    # ========== Generate signals ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_TREND = 0.35  # Trend following position
    SIZE_MR = 0.25     # Mean reversion position (smaller, more risky)
    SIZE_HALF = 0.15   # Half position after TP
    
    # Regime thresholds
    REGIME_TREND_THRESHOLD = 0.35  # BBW percentile < 35% = low vol = trend regime
    REGIME_MR_THRESHOLD = 0.65     # BBW percentile > 65% = high vol = mean revert regime
    
    # Entry thresholds
    RSI_MR_LONG = 35
    RSI_MR_SHORT = 65
    RSI_TREND_MIN = 45
    RSI_TREND_MAX = 60
    ATR_STOP_MULT = 2.0  # 2*ATR stoploss
    TP_MULT = 2.0        # 2R take profit
    
    first_valid = max(200, 100, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(kama_15m[i]):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ========== Regime Detection ==========
        bbw_percentile = bbw_pct_15m[i]
        
        # Determine regime
        if bbw_percentile < REGIME_TREND_THRESHOLD:
            regime = 'trend'      # Low volatility - trend following
        elif bbw_percentile > REGIME_MR_THRESHOLD:
            regime = 'mean_revert'  # High volatility - mean reversion
        else:
            regime = 'neutral'    # Middle - stay out or reduce position
        
        # ========== 4h Trend Filter ==========
        kama_trend_4h = 1 if close[i] > kama_4h_aligned[i] else (-1 if close[i] < kama_4h_aligned[i] else 0)
        st_trend_4h = st_4h_aligned[i]
        
        # 4h trend bias (need agreement for strong signal)
        trend_bias = 0
        if kama_trend_4h == 1 and st_trend_4h == 1:
            trend_bias = 1
        elif kama_trend_4h == -1 and st_trend_4h == -1:
            trend_bias = -1
        
        # ========== Check existing positions first ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            atr = atr_15m[i]
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R after TP
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R after TP
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
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
        
        # ========== New Entry Logic Based on Regime ==========
        rsi_val = rsi_15m[i]
        st_15m = st_direction_15m[i]
        kama_slope_15m = kama_15m[i] - kama_15m[i - 5] if i >= 5 else 0
        
        if regime == 'trend':
            # TREND REGIME: Follow 4h trend bias, enter on 15m confirmation
            
            # LONG: 4h bullish + 15m Supertrend bullish + KAMA sloping up
            if trend_bias == 1 and st_15m == 1 and kama_slope_15m > 0:
                # RSI not overbought
                if RSI_TREND_MIN <= rsi_val <= RSI_TREND_MAX:
                    signals[i] = SIZE_TREND
                    position_side[i] = 1
                    entry_price[i] = close[i]
                    tp_triggered[i] = False
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
            
            # SHORT: 4h bearish + 15m Supertrend bearish + KAMA sloping down
            elif trend_bias == -1 and st_15m == -1 and kama_slope_15m < 0:
                # RSI not oversold
                if (100 - RSI_TREND_MAX) <= rsi_val <= (100 - RSI_TREND_MIN):
                    signals[i] = -SIZE_TREND
                    position_side[i] = -1
                    entry_price[i] = close[i]
                    tp_triggered[i] = False
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
        
        elif regime == 'mean_revert':
            # MEAN REVERT REGIME: Fade extremes, smaller position size
            
            # LONG: RSI oversold + 4h trend not strongly bearish
            if rsi_val <= RSI_MR_LONG and trend_bias != -1:
                signals[i] = SIZE_MR
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            
            # SHORT: RSI overbought + 4h trend not strongly bullish
            elif rsi_val >= RSI_MR_SHORT and trend_bias != 1:
                signals[i] = -SIZE_MR
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        # Neutral regime: no new entries, let existing positions run or exit
        
        # Default: no position
        if signals[i] == 0:
            position_side[i] = 0
    
    return signals