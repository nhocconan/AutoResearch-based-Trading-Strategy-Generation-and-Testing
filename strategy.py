#!/usr/bin/env python3
"""
EXPERIMENT #008 - Donchian Breakout with RSI Pullback (30m Primary, 4h+1d HTF)
==================================================================================================
Hypothesis: Current best uses 4h+1d with HMA+RSI (Sharpe=0.537). This uses 30m+4h+1d with
Donchian channels + RSI pullback + volume confirmation. Donchian breakouts capture momentum
better than HMA crossovers, and 30m primary gives 2x more trade opportunities than 1h.

Key innovations:
1. 30m PRIMARY (not 1h): More entry opportunities, captures intraday momentum
2. 4h Donchian(20): Clean breakout-based trend filter (not MA lag)
3. 1d SMA(50): Regime filter - only long in bull market, short in bear
4. Volume confirmation: Avoid fake breakouts (volume > 20-period average)
5. RSI pullback entry: Enter on pullback within trend, not at breakout extreme

Why this should beat #005 (Sharpe=0.537):
- 30m timeframe = 2x more signals than 1h strategies
- Donchian channels = cleaner trend detection than HMA (no whipsaw in ranges)
- Daily regime filter = avoids counter-trend trades in strong markets
- Volume filter = reduces false breakouts significantly
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_rsi_volume_mtf_30m_4h_1d_v1"
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
    Donchian Channel - returns upper band, lower band, and mid line
    Trend direction: price above mid = uptrend, below mid = downtrend
    """
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    mid = (upper + lower) / 2
    
    return upper, lower, mid


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma


def calculate_volume_sma(volume, period=20):
    """Calculate volume moving average"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h Donchian channel for trend direction
        _, lower_4h, mid_4h = calculate_donchian(high_4h, low_4h, period=20)
        
        # 4h RSI for momentum confirmation
        rsi_4h = calculate_rsi(close_4h, period=14)
        
        # Align to 30m timeframe (auto shift for completed bars)
        mid_4h_aligned = align_htf_to_ltf(prices, df_4h, mid_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        
    except Exception:
        mid_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
    
    # ========== 1d INDICATORS (REGIME FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        
        # Daily SMA(50) for bull/bear regime
        sma_1d = calculate_sma(close_1d, period=50)
        
        # Align to 30m timeframe
        sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
        
    except Exception:
        sma_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20   # Base position (20% of capital)
    SIZE_HIGH = 0.30   # High conviction (30% of capital)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5  # Wider stop for 30m noise
    
    # RSI zones for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 65
    
    # Volume filter (avoid low volume breakouts)
    VOLUME_MULT = 1.2  # Volume must be > 1.2x average
    
    # Donchian trend threshold
    DONCHIAN_TREND_THRESHOLD = 0.005  # 0.5% above/below mid
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_30m[i]) or atr_30m[i] == 0 or np.isnan(rsi_30m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_30m[i]
        rsi_val = rsi_30m[i]
        vol_ratio = volume[i] / vol_sma_30m[i] if vol_sma_30m[i] > 0 else 0
        
        # 4h trend filters (MASTER FILTER)
        mid_4h = mid_4h_aligned[i]
        rsi_4h = rsi_4h_aligned[i]
        
        # 1d regime filter
        sma_1d = sma_1d_aligned[i]
        
        # Determine 4h trend direction from Donchian mid
        trend_4h = 0
        if mid_4h > 0 and price > mid_4h * (1 + DONCHIAN_TREND_THRESHOLD):
            trend_4h = 1
        elif mid_4h > 0 and price < mid_4h * (1 - DONCHIAN_TREND_THRESHOLD):
            trend_4h = -1
        
        # Determine daily regime
        regime_1d = 0
        if sma_1d > 0 and price > sma_1d:
            regime_1d = 1  # Bull market
        elif sma_1d > 0 and price < sma_1d:
            regime_1d = -1  # Bear market
        
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
            
            # Stoploss check (2.5*ATR)
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
                    signals[i] = SIZE_BASE  # Reduce to half
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
                    signals[i] = -SIZE_BASE  # Reduce to half
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
        
        # ========== ENTRY LOGIC - DONCHIAN TREND + RSI PULLBACK + VOLUME ==========
        # LONG: 4h trend up + daily bull + 30m RSI pullback + volume confirmation
        long_condition = (
            trend_4h == 1 and
            regime_1d >= 0 and  # Bull or neutral daily
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            vol_ratio >= VOLUME_MULT and
            rsi_4h > 45 and  # 4h momentum not oversold
            price > mid_4h  # Price above 4h Donchian mid
        )
        
        # SHORT: 4h trend down + daily bear + 30m RSI pullback + volume confirmation
        short_condition = (
            trend_4h == -1 and
            regime_1d <= 0 and  # Bear or neutral daily
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            vol_ratio >= VOLUME_MULT and
            rsi_4h < 55 and  # 4h momentum not overbought
            price < mid_4h  # Price below 4h Donchian mid
        )
        
        # Determine position size based on conviction
        # High conviction: All filters aligned strongly
        high_conviction_long = (
            long_condition and
            regime_1d == 1 and  # Strong bull daily
            rsi_4h > 50 and  # 4h momentum bullish
            vol_ratio > 1.5  # Strong volume
        )
        
        high_conviction_short = (
            short_condition and
            regime_1d == -1 and  # Strong bear daily
            rsi_4h < 50 and  # 4h momentum bearish
            vol_ratio > 1.5  # Strong volume
        )
        
        if long_condition:
            size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            size = SIZE_HIGH if high_conviction_short else SIZE_BASE
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