# Strategy: bollinger_squeeze_breakout

## Status
ACTIVE - Sharpe=-1.221 | Return=+1.3% | DD=-8.4%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.906 | -1.9% | -6.9% | 78 |
| ETHUSDT | -0.744 | +10.1% | -5.5% | 97 |
| SOLUSDT | -1.011 | -4.3% | -12.8% | 186 |

## Code
```python
#!/usr/bin/env python3
"""
strategy.py - Bollinger Squeeze Breakout with Trend Filter
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Volatility compression followed by expansion creates high-probability breakouts.
    - Detect Bollinger Band squeeze (low volatility regime)
    - Wait for price breakout with volume confirmation
    - Filter by EMA trend direction for bias
    - Use RSI to avoid overextended entries
    - ATR-based position sizing for risk management

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "bollinger_squeeze_breakout"
timeframe = "1h"
leverage = 2.5  # Moderate leverage with volatility scaling

# Strategy parameters
BB_PERIOD = 20                # Bollinger Bands period
BB_STD = 2.0                  # Bollinger Bands standard deviation
SQUEEZE_LOOKBACK = 10         # Lookback for squeeze detection
SQUEEZE_THRESHOLD = 0.6       # BB width ratio threshold for squeeze
EMA_FAST = 20                 # Fast EMA for trend filter
EMA_SLOW = 50                 # Slow EMA for trend filter
RSI_PERIOD = 14               # RSI calculation period
RSI_OVERBOUGHT = 75           # RSI overbought threshold
RSI_OVERSOLD = 25             # RSI oversold threshold
VOLUME_LOOKBACK = 20          # Lookback for volume average
VOLUME_THRESHOLD = 1.5        # Volume spike multiplier for breakout
ATR_PERIOD = 14               # ATR calculation period
VOLATILITY_TARGET = 0.015     # Target volatility for position sizing
MIN_SIGNAL = 0.15             # Minimum signal magnitude to trade
MAX_SIGNAL = 0.85             # Maximum signal magnitude


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    # Use pandas for efficient rolling calculation (past data only)
    close_series = pd.Series(close)
    sma_values = close_series.rolling(window=period, min_periods=period).mean().values
    sma = np.nan_to_num(sma_values, nan=0.0)
    
    return sma


def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    # Use pandas for efficient calculation (past data only)
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema = np.nan_to_num(ema_values, nan=0.0)
    
    return ema


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Number of standard deviations
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band, bb_width)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    width = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, width
    
    close_series = pd.Series(close)
    
    # Calculate middle band (SMA)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    
    # Calculate standard deviation
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    # Calculate bands
    upper_series = middle_series + (std_dev * std_series)
    lower_series = middle_series - (std_dev * std_series)
    
    # Calculate bandwidth (normalized width)
    width_series = (upper_series - lower_series) / middle_series
    
    upper = np.nan_to_num(upper_series.values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num(lower_series.values, nan=0.0)
    width = np.nan_to_num(width_series.values, nan=0.0)
    
    return upper, middle, lower, width


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    
    Args:
        close: Array of close prices
        period: RSI period
    
    Returns:
        Array of RSI values (0-100)
    """
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    close_series = pd.Series(close)
    
    # Calculate price changes
    delta = close_series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gains and losses using Wilder's smoothing
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.nan_to_num(rsi_series.values, nan=50.0)
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period
    
    Returns:
        Array of ATR values
    """
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Use pandas for Wilder's smoothing (equivalent to EMA with alpha=1/period)
    tr_series = pd.Series(tr)
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    atr = np.nan_to_num(atr_series.values, nan=0.0)
    
    return atr


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio relative to rolling average.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume ratios
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    # Avoid division by zero
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


def detect_squeeze(bb_width: np.ndarray, lookback: int = 10, threshold: float = 0.6) -> np.ndarray:
    """
    Detect Bollinger Band squeeze (low volatility regime).
    Squeeze occurs when BB width is below threshold relative to recent history.
    
    Args:
        bb_width: Array of Bollinger Band widths
        lookback: Lookback period for width comparison
        threshold: Threshold ratio for squeeze detection
    
    Returns:
        Array of squeeze indicators (1.0 = squeeze, 0.0 = no squeeze)
    """
    n = len(bb_width)
    squeeze = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return squeeze
    
    bb_width_series = pd.Series(bb_width)
    
    # Calculate rolling max width over lookback period
    rolling_max = bb_width_series.rolling(window=lookback, min_periods=lookback).max().values
    
    # Squeeze when current width is below threshold of recent max
    mask = rolling_max > 0
    squeeze[mask] = np.where(bb_width[mask] / rolling_max[mask] < threshold, 1.0, 0.0)
    
    return squeeze


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Bollinger Squeeze Breakout Strategy with Trend Filter.
    
    Signal Logic:
    1. Detect Bollinger Band squeeze (low volatility compression)
    2. Wait for price breakout above/below bands with volume confirmation
    3. Filter by EMA trend direction for bias
    4. Use RSI to avoid overextended entries
    5. ATR-based volatility scaling for position sizing
    
    Entry Conditions:
    - LONG: Squeeze detected + price breaks above upper BB + volume spike + 
            EMA fast > slow + RSI not overbought
    - SHORT: Squeeze detected + price breaks below lower BB + volume spike +
             EMA fast < slow + RSI not oversold
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract required columns with safety checks
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
    except (KeyError, TypeError, ValueError) as e:
        # Return zeros if required columns missing
        return signals
    
    # Handle any NaN values in price data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure no zero or negative prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD
    )
    
    # Detect squeeze conditions
    squeeze = detect_squeeze(bb_width, SQUEEZE_LOOKBACK, SQUEEZE_THRESHOLD)
    
    # Calculate EMAs for trend filter
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    # Calculate RSI for entry filtering
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate volume ratio
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index (all indicators need warmup period)
    min_valid_index = max(
        BB_PERIOD,
        SQUEEZE_LOOKBACK,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Track squeeze state for breakout detection
    in_squeeze = False
    squeeze_start_idx = -1
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0 or bb_width[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Update squeeze state
        if squeeze[i] == 1.0:
            if not in_squeeze:
                in_squeeze = True
                squeeze_start_idx = i
        else:
            in_squeeze = False
            squeeze_start_idx = -1
        
        # Check for breakout conditions
        # Need squeeze to have been active recently (within last SQUEEZE_LOOKBACK bars)
        recent_squeeze = False
        for j in range(max(min_valid_index, i - SQUEEZE_LOOKBACK), i + 1):
            if squeeze[j] == 1.0:
                recent_squeeze = True
                break
        
        if not recent_squeeze:
            signals[i] = 0.0
            continue
        
        # Trend direction from EMA relationship
        trend_bullish = ema_fast[i] > ema_slow[i]
        trend_bearish = ema_fast[i] < ema_slow[i]
        
        # RSI filter - avoid buying overbought, selling oversold
        rsi_bullish_ok = rsi[i] < RSI_OVERBOUGHT
        rsi_bearish_ok = rsi[i] > RSI_OVERSOLD
        
        # Volume confirmation for breakout
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Breakout detection
        # Long: price closes above upper band
        breakout_long = close[i] > bb_upper[i]
        # Short: price closes below lower band
        breakout_short = close[i] < bb_lower[i]
        
        # Calculate breakout strength (how far price is beyond band)
        breakout_strength_long = 0.0
        breakout_strength_short = 0.0
        
        if breakout_long and bb_upper[i] > 0:
            breakout_strength_long = (close[i] - bb_upper[i]) / bb_upper[i]
            breakout_strength_long = min(breakout_strength_long, 0.05)  # Cap at 5%
        
        if breakout_short and bb_lower[i] > 0:
            breakout_strength_short = (bb_lower[i] - close[i]) / bb_lower[i]
            breakout_strength_short = min(breakout_strength_short, 0.05)  # Cap at 5%
        
        # Calculate trend strength (EMA spread normalized by price)
        ema_spread = abs(ema_fast[i] - ema_slow[i]) / close[i]
        trend_strength = min(ema_spread * 100, 1.0)  # Cap at 1.0
        
        # Volatility adjustment (reduce position in high volatility)
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Scale inversely to volatility, target ~1.5% hourly volatility
            vol_factor = min(1.5, VOLATILITY_TARGET / max(atr_pct, 0.001))
        
        # RSI quality factor (better signals when RSI confirms trend)
        rsi_quality = 1.0
        if trend_bullish:
            # For longs, prefer RSI in 40-70 range (momentum but not overbought)
            if 40 <= rsi[i] <= 70:
                rsi_quality = 1.0
            elif rsi[i] < 40:
                rsi_quality = 0.7  # Weak momentum
            else:
                rsi_quality = 0.6  # Approaching overbought
        else:
            # For shorts, prefer RSI in 30-60 range
            if 30 <= rsi[i] <= 60:
                rsi_quality = 1.0
            elif rsi[i] > 60:
                rsi_quality = 0.7  # Weak momentum
            else:
                rsi_quality = 0.6  # Approaching oversold
        
        # Base signal from breakout direction
        raw_signal = 0.0
        signal_confidence = 0.0
        
        if breakout_long and trend_bullish and rsi_bullish_ok:
            # Long signal
            signal_confidence = breakout_strength_long * 20  # Scale to ~1.0
            if volume_confirmed:
                signal_confidence *= 1.3
            signal_confidence *= trend_strength
            raw_signal = signal_confidence
        elif breakout_short and trend_bearish and rsi_bearish_ok:
            # Short signal
            signal_confidence = breakout_strength_short * 20  # Scale to ~1.0
            if volume_confirmed:
                signal_confidence *= 1.3
            signal_confidence *= trend_strength
            raw_signal = -signal_confidence
        
        # Apply RSI quality factor
        raw_signal *= rsi_quality
        
        # Apply volatility adjustment
        signal = raw_signal * vol_factor
        
        # Apply minimum signal threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-MAX_SIGNAL, MAX_SIGNAL] to leave room for portfolio scaling
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals
```

## Last Updated
2026-03-20 19:56
