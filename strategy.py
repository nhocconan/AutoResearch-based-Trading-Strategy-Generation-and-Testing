#!/usr/bin/env python3
"""
EXPERIMENT #100 - MTF Supertrend+RSI+MACD+VolRegime (1h+4h Proper HTF v1)
==================================================================================================
Hypothesis: Recent ensemble failures (#090, #092) show complex voting creates churn and DD.
The winning strategy uses 15m+1h with clean trend+pullback logic.

Key changes from #040:
- Timeframe: 1h instead of 15m (less noise, fewer false signals, lower fees)
- MTF: 1h + 4h using PROPER mtf_data helper (critical - many failed without this)
- Add MACD momentum confirmation (proven in current best Sharpe=3.653)
- Volatility regime: use ATR percentile instead of BBW absolute (more adaptive)
- Adaptive sizing: reduce position in high volatility (ATR > 70th percentile)
- Cleaner entry logic: fewer filters, higher quality signals
- Stoploss: 2.5*ATR (slightly wider for 1h timeframe)
- Position size: 0.30 base, 0.20 in high vol (conservative)

Why this should beat #040 and recent failures:
- 1h has better signal quality than 15m (proven in multi-TF research)
- 4h trend filter via mtf_data is more reliable than synthetic resampling
- MACD adds momentum confirmation (in current best strategy)
- Volatility-adaptive sizing reduces DD in choppy markets
- Simpler logic = fewer bugs and less churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_macd_volregime_1h_4h_v1"
timeframe = "1h"
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


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
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


def calculate_volatility_percentile(atr, close, lookback=100):
    """Calculate ATR volatility percentile for regime detection"""
    n = len(close)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        rank = np.sum(window < atr[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    vol_pct = calculate_volatility_percentile(atr_1h, close, lookback=100)
    
    # 4h trend filter using PROPER mtf_data helper
    try:
        df_4h = get_htf_data(prices, '4h')
        if len(df_4h) > 0:
            close_4h = df_4h['close'].values
            high_4h = df_4h['high'].values
            low_4h = df_4h['low'].values
            
            # 4h Supertrend for trend direction
            st_4h, st_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
            
            # 4h RSI for momentum
            rsi_4h = calculate_rsi(close_4h, period=14)
            
            # Align 4h indicators to 1h timeframe (auto shift for completed bars)
            st_trend_4h = align_htf_to_ltf(prices, df_4h, st_dir_4h)
            rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        else:
            st_trend_4h = np.ones(n)
            rsi_4h_aligned = np.ones(n) * 50
    except Exception:
        # Fallback if mtf_data fails
        st_trend_4h = np.ones(n)
        rsi_4h_aligned = np.ones(n) * 50
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    SIZE_LOW_VOL = 0.35
    SIZE_HIGH_VOL = 0.20
    
    # Volatility regime thresholds
    VOL_LOW_THRESHOLD = 0.30  # Below 30th percentile = low vol
    VOL_HIGH_THRESHOLD = 0.70  # Above 70th percentile = high vol
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # MACD histogram threshold for momentum
    MACD_MIN = 0.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 100, 26 + 9, 14 * 2)
    
    # Track position state
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get 4h trend filter
        trend_4h = st_trend_4h[i] if not np.isnan(st_trend_4h[i]) else 1
        rsi_4h_val = rsi_4h_aligned[i] if not np.isnan(rsi_4h_aligned[i]) else 50
        
        # Get 1h indicators
        st_trend_1h = st_direction_1h[i]
        rsi_1h_val = rsi_1h[i]
        macd_val = macd_hist[i]
        atr = atr_1h[i]
        price = close[i]
        vol_regime = vol_pct[i]
        
        # Determine position size based on volatility regime
        if vol_regime < VOL_LOW_THRESHOLD:
            base_size = SIZE_LOW_VOL
        elif vol_regime > VOL_HIGH_THRESHOLD:
            base_size = SIZE_HIGH_VOL
        else:
            base_size = SIZE_FULL
        
        # 4h trend filter - only trade with higher timeframe trend
        if trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = -base_size * 0.5 if prev_side == -1 else base_size * 0.5
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -base_size * 0.5
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered - check if trend still valid
            if (prev_side == 1 and st_trend_1h == -1) or (prev_side == -1 and st_trend_1h == 1):
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h Supertrend + RSI pullback + MACD momentum
        if trend_4h == 1 and st_trend_1h == 1:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_1h_val <= RSI_LONG_MAX and 
                macd_val > MACD_MIN and
                rsi_4h_val >= 45):  # Pullback + momentum + 4h confirmation
                signals[i] = base_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and st_trend_1h == -1:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_1h_val <= RSI_SHORT_MAX and 
                macd_val < -MACD_MIN and
                rsi_4h_val <= 55):  # Pullback + momentum + 4h confirmation
                signals[i] = -base_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals