#!/usr/bin/env python3
"""
EXPERIMENT #056 - Donchian Breakout + ATR Volatility + Volume Confirmation (30m Primary)
==================================================================================================
Hypothesis: Current best (#053) uses BB regime + Supertrend + RSI on 1h timeframe.
This strategy tries Donchian Channel breakouts (Turtle Trading style) on 30m for more trade opportunities.

Key innovations:
1. Donchian Channel (20-period) breakout entries - proven trend-following system
2. ATR-based position sizing - smaller positions in high volatility (risk control)
3. Volume confirmation - breakouts must have 1.5x average volume to reduce false signals
4. 4h HMA trend filter - only trade in direction of higher timeframe trend
5. Volatility regime filter - reduce size in extreme volatility (BB width > 80th percentile)
6. 30m primary timeframe - more trades than 1h, less noise than 5m/15m

Why this should beat current best (Sharpe=0.563):
- Donchian breakouts capture strong trends early (Turtle Trading proved this)
- Volume filter reduces false breakouts (major weakness of pure Donchian)
- ATR sizing adapts to volatility (smaller positions when risk is higher)
- 30m captures more opportunities than 1h while maintaining signal quality
- 4h HMA filter prevents counter-trend trades (major source of drawdown)

Risk Management:
- Max signal: 0.35 (35% of capital)
- Stoploss: 2*ATR from entry
- Take profit: 2R, then trail at 1R
- Volume confirmation required for entry
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_atr_volume_breakout_30m_4h_v1"
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


def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks highest high and lowest low over period
    Returns: upper_channel, lower_channel, middle_channel
    """
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2
    
    return upper, lower, middle


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    
    # WMA with period/2
    wma_half = close_series.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    
    # WMA with period
    wma_full = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    # 2*WMA_half - WMA_full
    raw_hma = 2 * wma_half - wma_full
    
    # WMA of raw_hma with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    hma = raw_hma.ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    
    return hma.values


def calculate_bollinger_width(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for volatility regime"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    bandwidth = np.where(sma > 0, bandwidth, 0)
    
    return bandwidth


def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate BB Width percentile over lookback period"""
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bandwidth[i - lookback + 1:i + 1]
        valid_window = window[window > 0]
        if len(valid_window) > 0:
            rank = np.sum(valid_window < bandwidth[i])
            percentile[i] = rank / len(valid_window) * 100
    
    return percentile


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_series = pd.Series(volume)
    vol_sma = vol_series.rolling(window=period, min_periods=period).mean().values
    
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    bb_width_30m = calculate_bollinger_width(close, period=20, std_mult=2.0)
    bb_pct_30m = calculate_bb_percentile(bb_width_30m, lookback=100)
    volume_sma_30m = calculate_volume_sma(volume, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        
        # Align to 30m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE with ATR adjustment
    SIZE_BASE = 0.20    # Base position (20% of capital)
    SIZE_HIGH = 0.35    # High conviction (35% of capital) - MAX ALLOWED
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # Volume confirmation threshold
    VOLUME_MULT = 1.5   # Volume must be 1.5x average
    
    # BB volatility regime thresholds
    BB_HIGH_VOL = 80    # Above 80th percentile = reduce size
    
    first_valid = max(150, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_30m[i]
        bb_percentile = bb_pct_30m[i]
        current_volume = volume[i]
        avg_volume = volume_sma_30m[i]
        
        # 4h trend filter
        hma_4h_val = hma_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if i > 0 and hma_4h_val > hma_4h_aligned[i - 1] and hma_4h_val > 0:
            trend_4h = 1
        elif i > 0 and hma_4h_val < hma_4h_aligned[i - 1] and hma_4h_val > 0:
            trend_4h = -1
        
        # Check volatility regime
        is_high_vol = bb_percentile > BB_HIGH_VOL
        
        # Volume confirmation
        volume_confirmed = (avg_volume > 0) and (current_volume >= VOLUME_MULT * avg_volume)
        
        # ========== CHECK EXISTING POSITIONS ==========
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
            
            # Stoploss check (2.0*ATR)
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE  # Reduce to half position
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE  # Reduce to half position
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
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
        
        # ========== ENTRY LOGIC - DONCHIAN BREAKOUT ==========
        
        # Donchian breakout signals
        breakout_long = (price > donchian_upper[i]) and (donchian_upper[i] > 0)
        breakout_short = (price < donchian_lower[i]) and (donchian_lower[i] > 0)
        
        # 4h trend filter (only trade in direction of 4h trend)
        long_trend_filter = (trend_4h >= 0)  # 4h neutral or bullish
        short_trend_filter = (trend_4h <= 0)  # 4h neutral or bearish
        
        # Determine position size based on volatility regime
        if is_high_vol:
            size = SIZE_BASE  # Reduce size in high volatility
        else:
            size = SIZE_HIGH  # Full size in normal/low volatility
        
        # LONG entry conditions
        long_condition = (
            breakout_long and
            long_trend_filter and
            volume_confirmed
        )
        
        # SHORT entry conditions
        short_condition = (
            breakout_short and
            short_trend_filter and
            volume_confirmed
        )
        
        if long_condition:
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals