#!/usr/bin/env python3
"""
EXPERIMENT #099 - MTF Triple Confirmation with Volatility-Adaptive Sizing (15m+1h+4h)
==================================================================================================
Hypothesis: The best performing strategy (#040, Sharpe=5.4) used 4h HMA + 1h RSI + Z-score.
Current best (#098) uses Supertrend+RSI but lacks momentum confirmation. 

Key changes from #098:
- Add 1h MACD histogram as momentum confirmation (proven in best strategy)
- Triple timeframes: 4h trend + 1h momentum + 15m entry timing
- Volatility-adaptive position sizing: scale by ATR percentile AND signal agreement
- Tighter stoploss at 1.5*ATR (vs 2*ATR) with faster trailing at 1R
- Signal confidence scoring: more confirmations = larger position (max 0.35)

Why this should work:
- 4h Supertrend filters major trend direction
- 1h MACD confirms momentum alignment (reduces false entries)
- 15m RSI provides precise pullback entry timing
- Volatility scaling reduces position size in high-vol regimes (critical for DD control)
- Based on lessons from #040 (best ever) and #098 (current keeper)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_triple_confirm_vol_adaptive_15m_1h_4h_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = 1 if close[i] > supertrend[i] else -1
        else:
            if trend[i - 1] == 1:
                supertrend[i] = max(upper_band[i], supertrend[i - 1]) if upper_band[i] < supertrend[i - 1] else upper_band[i]
                if close[i] < supertrend[i]:
                    trend[i] = -1
                    supertrend[i] = lower_band[i]
            else:
                supertrend[i] = min(lower_band[i], supertrend[i - 1]) if lower_band[i] > supertrend[i - 1] else lower_band[i]
                if close[i] > supertrend[i]:
                    trend[i] = 1
                    supertrend[i] = upper_band[i]
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # EMA calculations
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    
    for i in range(slow + signal, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_sma(close, period=20):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    sma_15m = calculate_sma(close, period=200)
    
    # 1h indicators for momentum confirmation
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        macd_1h, _, hist_1h = calculate_macd(close_1h, fast=12, slow=26, signal=9)
        hist_1h_aligned = align_htf_to_ltf(prices, df_1h, hist_1h)
    except Exception:
        hist_1h_aligned = np.zeros(n)
    
    # 4h indicators for trend
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        _, trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
        
    except Exception:
        trend_4h_aligned = np.ones(n)
    
    # Generate signals with triple-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    MAX_SIZE = 0.35
    MED_SIZE = 0.20
    MIN_SIZE = 0.10
    
    # ATR percentile for position scaling
    atr_lookback = 100
    atr_percentile = np.zeros(n)
    for i in range(atr_lookback - 1, n):
        window = atr_15m[i - atr_lookback + 1:i + 1]
        rank = np.sum(window <= atr_15m[i])
        atr_percentile[i] = rank / atr_lookback
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier (tighter than #098)
    ATR_STOP_MULT = 1.5
    ATR_TRAIL_MULT = 1.0
    TP_MULT = 2.5
    
    first_valid = max(200, 14 * 2, 100, 35 + 9)
    
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
        
        # 4h Supertrend trend filter
        trend_4h_val = trend_4h_aligned[i]
        
        # 1h MACD momentum filter
        macd_1h_val = hist_1h_aligned[i]
        
        # Price vs 200 SMA filter
        price_vs_sma = 0
        if sma_15m[i] > 0:
            if close[i] > sma_15m[i]:
                price_vs_sma = 1
            elif close[i] < sma_15m[i]:
                price_vs_sma = -1
        
        # ATR-based position scaling
        atr_pct = atr_percentile[i]
        if atr_pct > 0.7:
            vol_scale = 0.5
        elif atr_pct > 0.4:
            vol_scale = 0.75
        else:
            vol_scale = 1.0
        
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        
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
            
            # Stoploss check (1.5*ATR)
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
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = signals[i - 1] * 0.5
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_TRAIL_MULT * atr
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
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = signals[i - 1] * 0.5
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_TRAIL_MULT * atr
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
        
        # Signal confidence scoring (0-3 confirmations)
        confidence = 0
        
        # Entry logic: Triple confirmation required
        # Long: 4h Supertrend bullish + 1h MACD positive + 15m RSI pullback + price above 200 SMA
        long_confirm = 0
        if trend_4h_val == 1:
            long_confirm += 1
        if macd_1h_val > 0:
            long_confirm += 1
        if price_vs_sma >= 0:
            long_confirm += 1
        
        if long_confirm >= 2 and (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
            confidence = long_confirm
        
        # Short: 4h Supertrend bearish + 1h MACD negative + 15m RSI pullback + price below 200 SMA
        short_confirm = 0
        if trend_4h_val == -1:
            short_confirm += 1
        if macd_1h_val < 0:
            short_confirm += 1
        if price_vs_sma <= 0:
            short_confirm += 1
        
        if short_confirm >= 2 and (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
            confidence = short_confirm
        
        # Position size based on confidence and volatility
        if confidence >= 3:
            position_size = MAX_SIZE * vol_scale
        elif confidence == 2:
            position_size = MED_SIZE * vol_scale
        else:
            position_size = 0.0
        
        # Execute entry
        if confidence >= 2 and long_confirm >= 2 and (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
            signals[i] = position_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif confidence >= 2 and short_confirm >= 2 and (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
            signals[i] = -position_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals