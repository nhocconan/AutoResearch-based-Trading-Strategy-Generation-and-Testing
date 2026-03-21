#!/usr/bin/env python3
"""
EXPERIMENT #022 - KAMA Trend + BB-RSI Entry + Volume Confirmation + Chandelier Exit
====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to volatility regimes than HMA.
Combine Bollinger Band position (price location within bands) with RSI for better entry timing.
Add volume confirmation to filter weak moves. Use Chandelier Exit for trailing stops.

Key improvements over #021:
- KAMA(14,2,30) instead of HMA - adapts efficiency ratio to market noise
- 1h timeframe instead of 15m - fewer false signals, cleaner trends
- BB position filter (price in lower 40% of bands for long) - better entry location
- Volume confirmation (volume > 1.5x SMA20) - confirms move strength
- Chandelier Exit (22-period, 3*ATR) for trailing - professional trend-following stop
- Regime-adjusted position sizing (reduce size in high volatility)

Why this might beat Sharpe=11.523:
- KAMA reduces whipsaws in choppy markets better than HMA
- BB position ensures we enter at better prices within the trend
- Volume filter avoids low-liquidity traps
- Chandelier Exit is proven trend-following stop used by professionals
- 1h timeframe balances signal quality vs opportunity frequency
"""

import numpy as np
import pandas as pd

name = "mtf_kama_bb_rsi_vol_chandelier_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average
    KAMA adapts to market noise using Efficiency Ratio
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama[period - 1] = close[period - 1]
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    
    # Calculate BB position (0 = lower band, 1 = upper band)
    bb_position = np.zeros(n)
    mask = (upper - lower) > 0
    bb_position[mask] = (close[mask] - lower[mask]) / (upper[mask] - lower[mask])
    
    return upper, lower, mean, bb_position


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def calculate_chandelier_exit(high, low, close, atr, period=22, mult=3.0):
    """
    Calculate Chandelier Exit for trailing stops
    Long exit: highest high - mult * ATR
    Short exit: lowest low + mult * ATR
    """
    n = len(close)
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        chandelier_long[i] = highest - mult * atr[i]
        chandelier_short[i] = lowest + mult * atr[i]
    
    return chandelier_long, chandelier_short


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, period=14, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bb_mean, bb_position = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    volume_sma = calculate_volume_sma(volume, period=20)
    chandelier_long, chandelier_short = calculate_chandelier_exit(high, low, close, atr_1h, period=22, mult=3.0)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h KAMA for trend
    kama_4h = calculate_kama(c_4h, period=14, fast_period=2, slow_period=30)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(30, len(c_4h)):
        kama_slope = kama_4h[i] - kama_4h[i - 5] if i >= 5 else 0
        
        if kama_4h[i] > kama_4h[i - 1] and c_4h[i] > kama_4h[i] and kama_slope > 0:
            trend_4h[i] = 1  # Bullish
        elif kama_4h[i] < kama_4h[i - 1] and c_4h[i] < kama_4h[i] and kama_slope < 0:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    SIZE_QUARTER = 0.10  # Quarter position (high volatility)
    
    # RSI thresholds for entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback
    RSI_SHORT_ENTRY = 55  # Enter short on rally
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # BB position thresholds
    BB_LONG_MAX = 0.40    # Enter long when price in lower 40% of BB
    BB_SHORT_MIN = 0.60   # Enter short when price in upper 40% of BB
    
    # Volume confirmation
    VOLUME_MULT = 1.5     # Volume must be > 1.5x SMA
    
    # ATR stoploss multiplier (Chandelier uses 3*ATR)
    ATR_STOP_MULT = 3.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.015  # Target 1.5% ATR for 1h
    
    first_valid = max(80, 30, 14, 20, 22)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    entry_atr = np.zeros(n)  # ATR at entry for R calculation
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_position[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_pos = bb_position[i]
        atr = atr_1h[i]
        price = close[i]
        vol_ratio = volume[i] / volume_sma[i] if volume_sma[i] > 0 else 1.0
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.04:  # ATR > 4% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # Check Chandelier Exit and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr
            
            if prev_side == 1:
                # Chandelier Exit for long
                chand_exit = chandelier_long[i]
                if price < chand_exit:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * prev_atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    entry_atr[i] = prev_atr
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
                    continue
                
            elif prev_side == -1:
                # Chandelier Exit for short
                chand_exit = chandelier_short[i]
                if price > chand_exit:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * prev_atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    entry_atr[i] = prev_atr
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
                    continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        # Adjust size based on volatility regime
        if current_atr_pct > 0.03:  # High volatility
            base_size = SIZE_QUARTER
        elif current_atr_pct > 0.02:  # Medium volatility
            base_size = SIZE_HALF
        else:  # Low volatility
            base_size = SIZE_FULL
        
        position_size = base_size * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_QUARTER, position_size))
        
        # Volume confirmation required for new entries
        volume_confirmed = vol_ratio >= VOLUME_MULT
        
        if trend == 1:  # 4h uptrend
            # BB-RSI entry in uptrend with volume confirmation
            if bb_pos < BB_LONG_MAX and rsi_val < RSI_LONG_ENTRY and rsi_val > 30 and volume_confirmed:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                entry_atr[i] = atr
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    entry_atr[i] = entry_atr[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # BB-RSI entry in downtrend with volume confirmation
            if bb_pos > BB_SHORT_MIN and rsi_val > RSI_SHORT_ENTRY and rsi_val < 70 and volume_confirmed:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                entry_atr[i] = atr
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    entry_atr[i] = entry_atr[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            entry_atr[i] = 0
    
    return signals