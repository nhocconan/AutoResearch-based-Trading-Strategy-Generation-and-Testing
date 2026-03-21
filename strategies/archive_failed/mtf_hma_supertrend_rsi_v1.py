#!/usr/bin/env python3
"""
EXPERIMENT #004 - HMA Trend + Supertrend Entry + RSI Pullback + ATR Sizing
===========================================================================
Hypothesis: HMA provides faster trend detection than KAMA/EMA with less lag.
Combined with Supertrend for entry confirmation and RSI pullback timing,
this should capture trends earlier while avoiding whipsaws. ATR-based
position sizing reduces exposure in high volatility regimes.

Key features:
- 4h HMA(21) slope for trend direction (faster than KAMA)
- 1h Supertrend(10, 3) for entry confirmation
- 1h RSI(14) pullback entries (45-55 range)
- Dynamic position sizing: base_size * (target_ATR / current_ATR)
- Trailing stop at 2*ATR from entry
- Discrete signal levels: 0.0, ±0.20, ±0.35

Why this might beat Sharpe=0.517:
- HMA reduces lag vs EMA/KAMA
- Supertrend filters false breakouts better than Donchian
- ATR sizing adapts to volatility regime
- Cleaner stoploss logic
"""

import numpy as np
import pandas as pd

name = "mtf_hma_supertrend_rsi_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        result = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            result[i] = np.sum(series[i - window + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_line, trend_direction (1=up, -1=down)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            if trend[i - 1] == 1:
                if close[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
            else:
                if close[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
    
    return supertrend, trend


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
    rsi[avg_loss == 0] = 100
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    supertrend_1h, supertrend_dir_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    # Calculate 4h HMA
    hma_4h = calculate_hma(c_4h, period=21)
    
    # 4h trend direction based on HMA slope and price position
    trend_4h = np.zeros(n_4h)
    for i in range(25, n_4h):
        hma_slope = hma_4h[i] - hma_4h[i - 3]  # 3-bar slope
        price_vs_hma = (c_4h[i] - hma_4h[i]) / hma_4h[i] if hma_4h[i] > 0 else 0
        
        if hma_slope > 0 and price_vs_hma > -0.02:
            trend_4h[i] = 1  # Bullish
        elif hma_slope < 0 and price_vs_hma < 0.02:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    BASE_SIZE = 0.30   # Base position size
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # ATR target for position sizing (normalize to ~2% of price)
    TARGET_ATR_PCT = 0.02
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 65    # Exit long when overbought
    RSI_EXIT_SHORT = 35   # Exit short when oversold
    
    # Supertrend confirmation
    SUPERTREND_CONFIRM = True
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 25 * 4)  # Wait for all indicators (4h HMA needs 25 bars * 4)
    
    # Track positions for trailing stop logic
    position_entry_price = np.zeros(n)
    position_direction = np.zeros(n)  # 1=long, -1=short, 0=none
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        supertrend_direction = supertrend_dir_1h[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        atr_pct = atr / price if price > 0 else 1.0
        if atr_pct > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            if position_direction[i - 1] != 0 and i > 0:
                position_direction[i] = 0  # Close position
            continue
        
        # Calculate dynamic position size based on ATR
        atr_ratio = TARGET_ATR_PCT / atr_pct if atr_pct > 0 else 1.0
        atr_ratio = np.clip(atr_ratio, 0.5, 1.5)  # Limit sizing adjustment
        dynamic_size = BASE_SIZE * atr_ratio
        
        # Check trailing stop for existing positions
        if i > 0 and position_direction[i - 1] != 0:
            entry_price = position_entry_price[i - 1]
            prev_dir = position_direction[i - 1]
            
            if prev_dir == 1:  # Long position
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    continue
                elif rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    continue
                else:
                    # Hold position, check if we should maintain
                    if trend == 1 and (not SUPERTREND_CONFIRM or supertrend_direction == 1):
                        signals[i] = signals[i - 1]
                        position_direction[i] = 1
                        position_entry_price[i] = entry_price
                    else:
                        signals[i] = 0.0
                        position_direction[i] = 0
                        continue
            elif prev_dir == -1:  # Short position
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    continue
                elif rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    continue
                else:
                    # Hold position
                    if trend == -1 and (not SUPERTREND_CONFIRM or supertrend_direction == -1):
                        signals[i] = signals[i - 1]
                        position_direction[i] = -1
                        position_entry_price[i] = entry_price
                    else:
                        signals[i] = 0.0
                        position_direction[i] = 0
                        continue
        
        # Entry logic
        if trend == 1:  # 4h uptrend
            if SUPERTREND_CONFIRM and supertrend_direction != 1:
                signals[i] = 0.0
                position_direction[i] = 0
                continue
            
            if rsi_val < RSI_LONG_ENTRY:
                # Pullback entry - full position
                signals[i] = dynamic_size * (SIZE_FULL / BASE_SIZE)
                signals[i] = np.clip(signals[i], 0, SIZE_FULL)
                position_direction[i] = 1
                position_entry_price[i] = price
            elif rsi_val < 50 and signals[i - 1] == 0:
                # Moderate pullback - half position
                signals[i] = dynamic_size * (SIZE_HALF / BASE_SIZE)
                signals[i] = np.clip(signals[i], 0, SIZE_HALF)
                position_direction[i] = 1
                position_entry_price[i] = price
            else:
                signals[i] = signals[i - 1] if i > 0 else 0.0
                position_direction[i] = position_direction[i - 1] if i > 0 else 0
                
        elif trend == -1:  # 4h downtrend
            if SUPERTREND_CONFIRM and supertrend_direction != -1:
                signals[i] = 0.0
                position_direction[i] = 0
                continue
            
            if rsi_val > RSI_SHORT_ENTRY:
                # Rally entry - full short
                signals[i] = -dynamic_size * (SIZE_FULL / BASE_SIZE)
                signals[i] = np.clip(signals[i], -SIZE_FULL, 0)
                position_direction[i] = -1
                position_entry_price[i] = price
            elif rsi_val > 50 and signals[i - 1] == 0:
                # Moderate rally - half short
                signals[i] = -dynamic_size * (SIZE_HALF / BASE_SIZE)
                signals[i] = np.clip(signals[i], -SIZE_HALF, 0)
                position_direction[i] = -1
                position_entry_price[i] = price
            else:
                signals[i] = signals[i - 1] if i > 0 else 0.0
                position_direction[i] = position_direction[i - 1] if i > 0 else 0
        else:  # No clear trend
            signals[i] = 0.0
            position_direction[i] = 0
    
    # Ensure discrete signal levels to reduce churn
    for i in range(1, n):
        if signals[i] != 0:
            if signals[i] > 0:
                signals[i] = SIZE_FULL if signals[i] >= 0.275 else SIZE_HALF
            else:
                signals[i] = -SIZE_FULL if signals[i] <= -0.275 else -SIZE_HALF
    
    return signals