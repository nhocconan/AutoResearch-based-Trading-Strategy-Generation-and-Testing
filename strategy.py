#!/usr/bin/env python3
"""
EXPERIMENT #004 - Multi-Timeframe EMA Crossover + RSI Pullback + ATR Regime
============================================================================
Hypothesis: 4h EMA(21/55) crossover provides cleaner trend signals than Supertrend,
combined with 1h RSI(14) pullback entries and ATR volatility regime filter for
dynamic position sizing. EMA crossover is more stable than Supertrend during
ranging markets, reducing whipsaw losses.

Key differences from mtf_supertrend_macd_adx_v1:
- EMA(21/55) crossover instead of Supertrend (smoother trend transitions)
- RSI pullback instead of MACD (better for mean-reversion within trends)
- ATR regime filter for dynamic sizing (reduce size in high volatility)
- Trailing stoploss via ATR (signal→0 when price moves 2*ATR against position)

Why this might beat Sharpe=1.768:
- EMA crossover has fewer false signals than Supertrend in choppy markets
- RSI pullback entries capture better risk/reward than MACD crosses
- ATR-based sizing reduces exposure during volatile periods (controls DD)
- Discrete signal levels (0.0, ±0.20, ±0.35) minimize fee churn
"""

import numpy as np
import pandas as pd

name = "mtf_ema_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    ema = np.zeros(n)
    multiplier = 2 / (period + 1)
    
    # Initialize with SMA
    if n >= period:
        ema[period - 1] = np.mean(close[:period])
        for i in range(period, n):
            ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


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


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_atr_pct(close, atr):
    """Calculate ATR as percentage of price (volatility regime)"""
    atr_pct = np.zeros(len(close))
    mask = close > 0
    atr_pct[mask] = (atr[mask] / close[mask]) * 100
    return atr_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk management
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_pct_1h = calculate_atr_pct(close, atr_1h)
    
    # 4h EMA crossover for trend filter (resample 1h → 4h)
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
    
    close_4h = df_4h['close'].values
    n_4h = len(close_4h)
    
    # Calculate 4h EMAs
    ema21_4h = calculate_ema(close_4h, period=21)
    ema55_4h = calculate_ema(close_4h, period=55)
    
    # Calculate 4h trend direction (EMA crossover)
    trend_4h = np.zeros(n_4h)
    for i in range(55, n_4h):
        if ema21_4h[i] > ema55_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif ema21_4h[i] < ema55_4h[i]:
            trend_4h[i] = -1  # Bearish
        else:
            trend_4h[i] = 0  # Neutral
    
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
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    SIZE_MIN = 0.10    # Minimum position in high volatility
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 65    # Exit long when RSI overbought
    RSI_EXIT_SHORT = 35   # Exit short when RSI oversold
    
    # ATR volatility regime thresholds (percentile-based)
    ATR_LOW = 1.5    # Low volatility - full size
    ATR_HIGH = 3.5   # High volatility - reduce size
    
    # Track entry prices for stoploss calculation
    entry_price = np.zeros(n)
    entry_signal = np.zeros(n)  # Track entry direction
    
    first_valid = max(55 * 4, 20, 14)  # Wait for all indicators (4h EMA55 needs 55*4 1h bars)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr_pct = atr_pct_1h[i]
        current_price = close[i]
        
        # Determine position size based on volatility regime
        if atr_pct < ATR_LOW:
            base_size = SIZE_FULL
        elif atr_pct < ATR_HIGH:
            base_size = SIZE_HALF
        else:
            base_size = SIZE_MIN
        
        # Check for stoploss (2*ATR against position)
        if entry_signal[i - 1] != 0 and i > 0:
            if entry_signal[i - 1] > 0:  # Long position
                stoploss_price = entry_price[i - 1] - 2 * atr_1h[i - 1]
                if current_price < stoploss_price:
                    signals[i] = 0.0
                    entry_signal[i] = 0.0
                    entry_price[i] = 0.0
                    continue
            elif entry_signal[i - 1] < 0:  # Short position
                stoploss_price = entry_price[i - 1] + 2 * atr_1h[i - 1]
                if current_price > stoploss_price:
                    signals[i] = 0.0
                    entry_signal[i] = 0.0
                    entry_price[i] = 0.0
                    continue
        
        # Generate new signals based on trend and RSI
        if trend == 1:  # 4h uptrend (EMA21 > EMA55)
            if rsi_val < RSI_LONG_ENTRY:
                # Pullback entry - go long
                signals[i] = base_size
                if entry_signal[i - 1] <= 0:  # New long entry
                    entry_price[i] = current_price
                    entry_signal[i] = 1
                else:
                    entry_price[i] = entry_price[i - 1]
                    entry_signal[i] = 1
            elif rsi_val > RSI_EXIT_LONG:
                # RSI overbought - exit long
                signals[i] = 0.0
                entry_signal[i] = 0.0
                entry_price[i] = 0.0
            else:
                # Hold existing position or stay flat
                signals[i] = signals[i - 1]
                entry_signal[i] = entry_signal[i - 1]
                entry_price[i] = entry_price[i - 1]
                
        elif trend == -1:  # 4h downtrend (EMA21 < EMA55)
            if rsi_val > RSI_SHORT_ENTRY:
                # Rally entry - go short
                signals[i] = -base_size
                if entry_signal[i - 1] >= 0:  # New short entry
                    entry_price[i] = current_price
                    entry_signal[i] = -1
                else:
                    entry_price[i] = entry_price[i - 1]
                    entry_signal[i] = -1
            elif rsi_val < RSI_EXIT_SHORT:
                # RSI oversold - exit short
                signals[i] = 0.0
                entry_signal[i] = 0.0
                entry_price[i] = 0.0
            else:
                # Hold existing position or stay flat
                signals[i] = signals[i - 1]
                entry_signal[i] = entry_signal[i - 1]
                entry_price[i] = entry_price[i - 1]
        else:  # No clear trend (EMA flat or crossing)
            signals[i] = 0.0
            entry_signal[i] = 0.0
            entry_price[i] = 0.0
    
    return signals