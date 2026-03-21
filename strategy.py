#!/usr/bin/env python3
"""
EXPERIMENT #116 - MTF Supertrend+MACD+RSI+BBW+Chandelier+VolAdj (15m+4h Proper HTF v1)
==================================================================================================
Hypothesis: Current best (mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1, Sharpe=3.653) uses 15m+1h+4h.
Recent failures (#104-#115) all had massive drawdowns due to poor position sizing and stop management.

Key improvements from current best:
- Use mtf_data helper for PROPER 4h alignment (get_htf_data, align_htf_to_ltf)
- Add Chandelier Exit (highest_high - 3*ATR(22)) for trailing stops
- Volatility-adjusted position sizing: size = base_size * (target_vol / current_vol)
- Keep 15m entries with 4h trend filter (proven combination)
- Discrete signal levels: 0.0, ±0.20, ±0.35 (max 0.40)
- Stoploss: 2.5*ATR (slightly wider than #040's 2.0*ATR to reduce whipsaws)
- Take profit: 2R with trail at 1R

Why this should beat current best:
- Proper HTF alignment using mtf_data (46 strategies failed without this)
- Chandelier exit provides better trailing than fixed ATR stops
- Volatility-adjusted sizing reduces exposure during high-vol regimes
- Based on proven winning indicators (Supertrend+MACD+RSI+BBW)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_macd_rsi_bbw_chandelier_voladj_15m_4h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    for i in range(n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
    
    return upper, middle, lower, bbw


def calculate_chandelier_exit(high, low, close, period=22, multiplier=3.0):
    """Calculate Chandelier Exit (ATR trailing stop)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    long_stop = np.zeros(n)
    short_stop = np.zeros(n)
    
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        long_stop[i] = highest[i] - multiplier * atr[i]
        short_stop[i] = lowest[i] + multiplier * atr[i]
    
    return long_stop, short_stop


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(high, low, close, period=22, multiplier=3.0)
    
    # Get 4h data using mtf_data helper (PROPER alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend
        hma_4h = calculate_hma(close_4h, period=21)
        supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        macd_4h, macd_signal_4h, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.zeros(n)
        macd_hist_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    SIZE_HALF = 0.15
    
    # Volatility-adjusted sizing parameters
    TARGET_VOL = 0.02  # Target 2% daily vol
    VOL_LOOKBACK = 20
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # MACD histogram threshold
    MACD_MIN = 0
    
    # BBW minimum for regime filter (4h)
    BBW_MIN = 0.015
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 40, 26 + 9, 22)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filters
        trend_4h = 0
        if close[i] > hma_4h_aligned[i] and hma_4h_aligned[i] > 0:
            trend_4h = 1
        elif close[i] < hma_4h_aligned[i] and hma_4h_aligned[i] > 0:
            trend_4h = -1
        
        st_trend_4h = st_direction_4h_aligned[i]
        macd_4h_val = macd_hist_4h_aligned[i]
        bbw_4h_val = bbw_4h_aligned[i]
        
        # 15m indicators
        st_trend_15m = st_direction_15m[i]
        rsi_val = rsi_15m[i]
        macd_hist_val = macd_hist_15m[i]
        atr = atr_15m[i]
        price = close[i]
        
        # Calculate volatility-adjusted position size
        if i >= VOL_LOOKBACK:
            returns = np.diff(close[i-VOL_LOOKBACK:i+1]) / close[i-VOL_LOOKBACK:i]
            current_vol = np.std(returns) * np.sqrt(96)  # Annualize for 15m (96 bars/day)
            if current_vol > 0:
                vol_adjustment = min(1.5, max(0.5, TARGET_VOL / current_vol))
            else:
                vol_adjustment = 1.0
        else:
            vol_adjustment = 1.0
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (4h HMA + Supertrend + MACD)
        if trend_4h != st_trend_4h or trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # MACD histogram must confirm trend direction (4h)
        if trend_4h == 1 and macd_4h_val < MACD_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        if trend_4h == -1 and macd_4h_val > -MACD_MIN:
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
            
            # Chandelier Exit stoploss check
            if prev_side == 1:
                chandelier_stop = chandelier_long_15m[i]
                fixed_stop = prev_entry - ATR_STOP_MULT * atr
                stoploss_price = max(chandelier_stop, fixed_stop)
                
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
                    signals[i] = SIZE_HALF * vol_adjustment
                    position_side[i] = 1
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
                chandelier_stop = chandelier_short_15m[i]
                fixed_stop = prev_entry + ATR_STOP_MULT * atr
                stoploss_price = min(chandelier_stop, fixed_stop)
                
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
                    signals[i] = -SIZE_HALF * vol_adjustment
                    position_side[i] = -1
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 15m Supertrend + RSI pullback + MACD confirmation
        base_signal = BASE_SIZE * vol_adjustment
        base_signal = min(0.40, max(-0.40, base_signal))  # Cap at 0.40
        
        if trend_4h == 1 and st_trend_4h == 1 and st_trend_15m == 1:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                macd_hist_val > MACD_MIN):  # Pullback + MACD confirmation
                signals[i] = base_signal
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and st_trend_4h == -1 and st_trend_15m == -1:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                macd_hist_val < -MACD_MIN):  # Pullback + MACD confirmation
                signals[i] = -base_signal
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals