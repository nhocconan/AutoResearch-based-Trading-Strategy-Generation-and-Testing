#!/usr/bin/env python3
"""
EXPERIMENT #015 - Supertrend + RSI Pullback + Volume Confirmation + ATR Stop
=============================================================================
Hypothesis: Supertrend provides cleaner trend signals than moving averages with less lag.
Combined with RSI pullback entries, volume confirmation, and ATR trailing stops,
this should capture strong trends while filtering false breakouts.

Key differences from mtf_keltner_rsi_adx_v1:
- Supertrend(10,3) for trend direction instead of Keltner channels
- Volume spike confirmation (1.5x 20-period avg) to validate entries
- Tighter RSI thresholds (40/60 instead of 45/55) for better pullback timing
- ATR trailing stop at 2.0*ATR (tighter risk management)

Why this might beat Sharpe=4.452:
- Supertrend adapts better to volatility regimes than Keltner
- Volume filter reduces false breakout entries significantly
- Tighter stops preserve capital during reversals
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_rsi_volume_v1"
timeframe = "1h"
leverage = 1.0


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
            trend[i] = 1
        else:
            # Supertrend logic
            if close[i] > supertrend[i - 1]:
                supertrend[i] = min(lower_band[i], supertrend[i - 1] if trend[i - 1] == 1 else lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = max(upper_band[i], supertrend[i - 1] if trend[i - 1] == -1 else upper_band[i])
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


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def calculate_rvi(high, low, close, volume, period=14):
    """
    Calculate Relative Volume Index (RVI)
    Measures volume strength relative to recent average
    """
    n = len(close)
    vol_sma = calculate_volume_sma(volume, period)
    
    rvi = np.zeros(n)
    mask = vol_sma > 0
    rvi[mask] = volume[mask] / vol_sma[mask]
    
    return rvi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    rvi_1h = calculate_rvi(high, low, close, volume, period=20)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
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
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend
    supertrend_4h, trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h >= 10:  # Wait for supertrend to initialize
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position with all confirmations
    SIZE_HALF = 0.20   # Reduced position with partial confirmations
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 40   # Enter long on deeper pullback in uptrend
    RSI_SHORT_ENTRY = 60  # Enter short on stronger rally in downtrend
    RSI_EXIT = 50         # Exit when RSI crosses middle
    
    # Volume confirmation thresholds
    VOL_MIN = 1.2         # Minimum volume ratio for entry confirmation
    VOL_STRONG = 1.5      # Strong volume for full position
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(80, 14, 20)  # Wait for all indicators
    
    # Track entry prices and stops for trailing logic
    entry_price = np.zeros(n)
    stop_price = np.zeros(n)
    position_type = np.zeros(n)  # 1=long, -1=short, 0=none
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(rvi_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        rvi_val = rvi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.04:  # ATR > 4% of price = too volatile
            # Check if we have existing position - apply stoploss
            if position_type[i - 1] != 0 and i > 0:
                if position_type[i - 1] == 1:  # Long
                    if price < stop_price[i - 1]:
                        signals[i] = 0.0
                        position_type[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_type[i] = position_type[i - 1]
                else:  # Short
                    if price > stop_price[i - 1]:
                        signals[i] = 0.0
                        position_type[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_type[i] = position_type[i - 1]
            else:
                signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend - look for long entries
            if rsi_val < RSI_LONG_ENTRY:
                # Pullback entry - check volume confirmation
                if rvi_val >= VOL_STRONG:
                    signals[i] = SIZE_FULL
                    entry_price[i] = price
                    stop_price[i] = price - ATR_STOP_MULT * atr
                    position_type[i] = 1
                elif rvi_val >= VOL_MIN:
                    signals[i] = SIZE_HALF
                    entry_price[i] = price
                    stop_price[i] = price - ATR_STOP_MULT * atr
                    position_type[i] = 1
                else:
                    # No volume confirmation - hold or exit
                    if i > 0 and position_type[i - 1] == 1:
                        if price < stop_price[i - 1]:
                            signals[i] = 0.0
                            position_type[i] = 0
                        else:
                            signals[i] = signals[i - 1]
                            position_type[i] = 1
                            # Trail stop higher
                            new_stop = price - ATR_STOP_MULT * atr
                            stop_price[i] = max(stop_price[i - 1], new_stop)
                    else:
                        signals[i] = 0.0
                        position_type[i] = 0
            elif rsi_val > RSI_EXIT and position_type[i - 1] == 1:
                # RSI crossed above 50 - reduce or exit
                if rsi_val > 70:  # Overbought - take profit
                    signals[i] = 0.0
                    position_type[i] = 0
                else:
                    signals[i] = signals[i - 1]
                    position_type[i] = 1
                    # Trail stop
                    new_stop = price - ATR_STOP_MULT * atr
                    stop_price[i] = max(stop_price[i - 1], new_stop)
            elif i > 0 and position_type[i - 1] == 1:
                # Hold existing long - check stoploss
                if price < stop_price[i - 1]:
                    signals[i] = 0.0
                    position_type[i] = 0
                else:
                    signals[i] = signals[i - 1]
                    position_type[i] = 1
                    # Trail stop higher
                    new_stop = price - ATR_STOP_MULT * atr
                    stop_price[i] = max(stop_price[i - 1], new_stop)
            else:
                signals[i] = 0.0
                position_type[i] = 0
                
        elif trend == -1:  # 4h downtrend - look for short entries
            if rsi_val > RSI_SHORT_ENTRY:
                # Rally entry - check volume confirmation
                if rvi_val >= VOL_STRONG:
                    signals[i] = -SIZE_FULL
                    entry_price[i] = price
                    stop_price[i] = price + ATR_STOP_MULT * atr
                    position_type[i] = -1
                elif rvi_val >= VOL_MIN:
                    signals[i] = -SIZE_HALF
                    entry_price[i] = price
                    stop_price[i] = price + ATR_STOP_MULT * atr
                    position_type[i] = -1
                else:
                    # No volume confirmation - hold or exit
                    if i > 0 and position_type[i - 1] == -1:
                        if price > stop_price[i - 1]:
                            signals[i] = 0.0
                            position_type[i] = 0
                        else:
                            signals[i] = signals[i - 1]
                            position_type[i] = -1
                            # Trail stop lower
                            new_stop = price + ATR_STOP_MULT * atr
                            stop_price[i] = min(stop_price[i - 1], new_stop)
                    else:
                        signals[i] = 0.0
                        position_type[i] = 0
            elif rsi_val < RSI_EXIT and position_type[i - 1] == -1:
                # RSI crossed below 50 - reduce or exit
                if rsi_val < 30:  # Oversold - take profit
                    signals[i] = 0.0
                    position_type[i] = 0
                else:
                    signals[i] = signals[i - 1]
                    position_type[i] = -1
                    # Trail stop
                    new_stop = price + ATR_STOP_MULT * atr
                    stop_price[i] = min(stop_price[i - 1], new_stop)
            elif i > 0 and position_type[i - 1] == -1:
                # Hold existing short - check stoploss
                if price > stop_price[i - 1]:
                    signals[i] = 0.0
                    position_type[i] = 0
                else:
                    signals[i] = signals[i - 1]
                    position_type[i] = -1
                    # Trail stop lower
                    new_stop = price + ATR_STOP_MULT * atr
                    stop_price[i] = min(stop_price[i - 1], new_stop)
            else:
                signals[i] = 0.0
                position_type[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_type[i] = 0
    
    return signals