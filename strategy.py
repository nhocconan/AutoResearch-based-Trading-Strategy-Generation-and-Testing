#!/usr/bin/env python3
"""
EXPERIMENT #023 - HMA Trend + RSI Pullback + BB Squeeze Filter + Volume Confirm
====================================================================================
Hypothesis: Building on #021's success (Sharpe=11.5), add Bollinger Band squeeze filter
to avoid choppy/low-volatility regimes where trend strategies fail. Add volume confirmation
to ensure entries have momentum behind them. Use 1h RSI instead of 15m for fewer false signals.
Tighter position sizing (0.30 max) and tighter stoploss (1.5*ATR) to reduce drawdown.

Key improvements over #021:
- Bollinger Band squeeze filter (BBW < 20th percentile = avoid trading)
- Volume confirmation (volume > 1.2 * 20-period avg volume)
- 1h RSI entries instead of 15m - fewer whipsaws, cleaner signals
- Tighter stoploss: 1.5*ATR instead of 2.0*ATR
- Lower max position size: 0.30 instead of 0.35
- Add MACD histogram confirmation for momentum

Why this might beat Sharpe=11.5:
- BB squeeze filter avoids choppy markets where HMA whipsaws
- Volume confirmation ensures entries have real momentum
- 1h RSI reduces noise vs 15m while still catching pullbacks
- Tighter risk management reduces drawdown during adverse moves
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_bb_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average - reduces lag vs EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    # Calculate WMA helper
    def wma(arr, w):
        result = np.zeros(len(arr))
        weights = np.arange(1, w + 1, dtype=float)
        weight_sum = np.sum(weights)
        for i in range(w - 1, len(arr)):
            if np.isnan(arr[i - w + 1:i + 1]).any():
                result[i] = np.nan
            else:
                result[i] = np.sum(arr[i - w + 1:i + 1] * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma


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
    if n >= period:
        atr[period - 1] = np.mean(tr[1:period])
        
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = (avg_loss > 0) & (avg_loss != 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[avg_loss == 0] = 100  # All gains
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Bandwidth"""
    n = len(close)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    bandwidth = (upper - lower) / mean if np.any(mean != 0) else np.zeros(n)
    
    # Handle division by zero
    bandwidth = np.where(mean != 0, (upper - lower) / mean, 0)
    
    return upper, lower, bandwidth


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    hma_16_1h = calculate_hma(close, period=16)
    hma_48_1h = calculate_hma(close, period=48)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    volume_sma = calculate_volume_sma(volume, period=20)
    
    # Calculate BB width percentile for squeeze filter
    bb_width_pct = np.zeros(n)
    valid_mask = ~np.isnan(bb_width)
    if np.any(valid_mask):
        for i in range(n):
            if i >= 100 and valid_mask[i]:
                # Rolling percentile of BB width over last 100 bars
                window_start = max(0, i - 99)
                valid_window = bb_width[window_start:i+1][~np.isnan(bb_width[window_start:i+1])]
                if len(valid_window) > 0:
                    bb_width_pct[i] = np.sum(valid_window <= bb_width[i]) / len(valid_window)
    
    # 4h HMA for trend filter (resample 1h → 4h)
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
    n_4h = len(c_4h)
    
    # Calculate 4h HMA for trend
    hma_16_4h = calculate_hma(c_4h, period=16)
    hma_48_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction based on HMA cross and price position
    trend_4h = np.zeros(n_4h)
    for i in range(48, n_4h):
        if np.isnan(hma_16_4h[i]) or np.isnan(hma_48_4h[i]):
            continue
        if hma_16_4h[i] > hma_48_4h[i] and c_4h[i] > hma_16_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif hma_16_4h[i] < hma_48_4h[i] and c_4h[i] < hma_16_4h[i]:
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
    SIZE_FULL = 0.30   # Full position (reduced from 0.35)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # BB squeeze filter - avoid low volatility regimes
    BB_SQUEEZE_THRESHOLD = 0.20  # Bottom 20% of BB width = squeeze, avoid trading
    
    # Volume confirmation
    VOLUME_MULT = 1.2  # Volume must be > 1.2x average
    
    # ATR stoploss multiplier (tighter than #021)
    ATR_STOP_MULT = 1.5
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # MACD histogram confirmation
    MACD_MIN = 0  # MACD histogram must be positive for longs
    
    first_valid = max(100, 48, 14, 20)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    entry_atr = np.zeros(n)  # Track ATR at entry for R calculation
    
    for i in range(first_valid, n):
        # Check for NaN values
        if (np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or 
            np.isnan(hma_16_1h[i]) or np.isnan(hma_48_1h[i]) or
            np.isnan(bb_width[i]) or np.isnan(macd_hist[i])):
            signals[i] = 0.0
            if i > 0:
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol = volume[i]
        vol_avg = volume_sma[i]
        bb_pct = bb_width_pct[i]
        macd_histogram = macd_hist[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.06:  # ATR > 6% of price = too volatile
            if position_side[i - 1] != 0 and i > 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
            else:
                signals[i] = 0.0
            continue
        
        # BB squeeze filter - avoid low volatility regimes
        if bb_pct < BB_SQUEEZE_THRESHOLD:
            # In squeeze, only hold existing positions, don't enter new
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Trailing stoploss (use entry ATR for consistency)
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * prev_atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    entry_atr[i] = prev_atr
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
                    entry_atr[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Trailing stoploss
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    entry_atr[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * prev_atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    entry_atr[i] = prev_atr
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
                    entry_atr[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Volume confirmation for new entries
        volume_confirmed = (vol_avg > 0) and (vol > VOLUME_MULT * vol_avg)
        
        # MACD confirmation
        macd_confirmed_long = macd_histogram > MACD_MIN
        macd_confirmed_short = macd_histogram < -MACD_MIN
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry in uptrend with volume and MACD confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30 and volume_confirmed and macd_confirmed_long:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                entry_atr[i] = atr
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    entry_atr[i] = entry_atr[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry in downtrend with volume and MACD confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70 and volume_confirmed and macd_confirmed_short:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                entry_atr[i] = atr
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    entry_atr[i] = entry_atr[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            entry_atr[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals