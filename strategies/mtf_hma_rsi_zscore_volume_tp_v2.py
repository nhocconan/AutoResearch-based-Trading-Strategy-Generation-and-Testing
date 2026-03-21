#!/usr/bin/env python3
"""
EXPERIMENT #024 - HMA Trend + RSI Pullback + Z-score Regime + Volume Confirmation
====================================================================================
Hypothesis: Building on #021's success (Sharpe=11.5), add Z-score regime filter and volume
confirmation to reduce false entries. Switch from 15m to 1h for entries (less noise, fewer
whipsaws). Use 4h HMA for trend + 1h RSI pullback + Z-score to avoid extremes + volume spike
confirmation on entries.

Key improvements over #021:
- 1h entries instead of 15m - fewer false signals, better signal-to-noise ratio
- Z-score regime filter - avoid entering when price > 2.5 std dev from mean
- Volume confirmation - require volume > 1.5x 20-period average on entry
- Tighter take profit at 1.5R instead of 2R - lock gains faster
- Trailing stop activates after 1R profit - protect gains
- Position size: 0.30 instead of 0.35 - slightly more conservative

Why this might beat Sharpe=11.5:
- 1h timeframe has fewer whipsaws than 15m while still catching moves
- Z-score filter avoids buying tops/selling bottoms
- Volume confirmation ensures real momentum behind entries
- Faster profit-taking reduces giveback on reversals
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_zscore_volume_tp_v2"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average - reduces lag vs EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    if half < 1:
        half = 1
    
    # Calculate WMA helper
    def wma(arr, w):
        result = np.zeros(len(arr))
        weights = np.arange(1, w + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(w - 1, len(arr)):
            result[i] = np.sum(arr[i - w + 1:i + 1] * weights) / w_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
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
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = (avg_loss > 0) & (avg_gain >= 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    hma_16_1h = calculate_hma(close, period=16)
    hma_48_1h = calculate_hma(close, period=48)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
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
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h HMA for trend
    hma_16_4h = calculate_hma(c_4h, period=16)
    hma_48_4h = calculate_hma(c_4h, period=48)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # 4h trend direction based on HMA cross and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(48, len(c_4h)):
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
    
    # Map 4h ATR to 1h (divide by 2 for approximate 1h ATR from 4h)
    atr_4h_mapped = np.zeros(n)
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(atr_4h):
            atr_4h_mapped[i] = atr_4h[idx_4h] / 2.0
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position (slightly more conservative than #021)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.0      # Don't enter if price > 2.0 std dev from mean (tighter than #021)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0   # Same as #021
    
    # Take profit multiplier (1.5R instead of 2R - lock gains faster)
    TP_MULT = 1.5
    
    # Trailing stop activation (after 1R profit)
    TRAIL_ACTIVATION_MULT = 1.0
    
    # Volume confirmation threshold
    VOLUME_MULT = 1.5     # Require volume > 1.5x 20-period average
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    first_valid = max(80, 48, 14, 20)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    trail_active = np.zeros(n)  # Track if trailing stop is active
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i] if atr_1h[i] > 0 else atr_4h_mapped[i]
        if atr <= 0:
            atr = 0.02 * close[i]  # Fallback to 2% of price
        price = close[i]
        vol = volume[i]
        vol_avg = vol_sma_1h[i]
        
        # Volume confirmation check
        volume_ok = (vol_avg > 0) and (vol > VOLUME_MULT * vol_avg)
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            trail_active[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_trail = trail_active[i - 1]
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Check if trailing stop should activate (after 1R profit)
                if not prev_trail and (price - prev_entry) >= TRAIL_ACTIVATION_MULT * ATR_STOP_MULT * atr:
                    trail_active[i] = 1
                    prev_trail = 1
                
                # Stoploss check
                if prev_trail:
                    # Trail stop at highest - 1.5*ATR after activation
                    stoploss_price = highest_since_entry[i] - 1.5 * atr
                else:
                    stoploss_price = prev_entry - ATR_STOP_MULT * atr
                
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trail_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (1.5R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    trail_active[i] = 1  # Activate trailing after TP
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trail_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Check if trailing stop should activate (after 1R profit)
                if not prev_trail and (prev_entry - price) >= TRAIL_ACTIVATION_MULT * ATR_STOP_MULT * atr:
                    trail_active[i] = 1
                    prev_trail = 1
                
                # Stoploss check
                if prev_trail:
                    # Trail stop at lowest + 1.5*ATR after activation
                    stoploss_price = lowest_since_entry[i] + 1.5 * atr
                else:
                    stoploss_price = prev_entry + ATR_STOP_MULT * atr
                
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trail_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (1.5R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    trail_active[i] = 1  # Activate trailing after TP
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trail_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Z-score filter - avoid entering at extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                trail_active[i] = trail_active[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))  # Clamp
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry in uptrend with volume confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30 and volume_ok:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                trail_active[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    trail_active[i] = trail_active[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trail_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry in downtrend with volume confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70 and volume_ok:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                trail_active[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    trail_active[i] = trail_active[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trail_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            trail_active[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals