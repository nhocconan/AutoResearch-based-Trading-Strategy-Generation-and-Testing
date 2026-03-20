# Strategy: adaptive_regime_trend_v1

## Status
ACTIVE - Sharpe=0.497 | Return=+652.8% | DD=-77.9%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.136 | -64.1% | -80.8% | 6534 |
| ETHUSDT | 0.418 | +32.8% | -72.1% | 7114 |
| SOLUSDT | 1.210 | +1989.7% | -80.7% | 7793 |

## Code
```python
#!/usr/bin/env python3
"""
strategy.py - Adaptive Regime Trend V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on trend_momentum_v2 success (Sharpe=0.330), improving:
    - Market regime detection (ADX-based trending vs ranging)
    - Adaptive signal logic based on regime
    - Better RSI divergence detection
    - Improved volatility-adjusted position sizing
    - Cleaner signal smoothing with hysteresis
    
    Key improvements over trend_momentum_v2:
    - ADX regime filter to avoid choppy markets
    - Different logic for trending vs ranging regimes
    - RSI divergence detection for early reversal signals
    - Dynamic signal thresholds based on volatility
    - Hysteresis to reduce signal flipping

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

name = "adaptive_regime_trend_v1"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_EXTREME_HIGH = 80
RSI_EXTREME_LOW = 20

# ADX regime detection
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25  # ADX above this = trending market
ADX_STRONG_THRESHOLD = 40  # ADX above this = strong trend

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_PERCENTILE_THRESHOLD = 0.5

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.25
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.6
HYSTERESIS_THRESHOLD = 0.08  # Minimum change to flip signal direction


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema = np.nan_to_num(ema_values, nan=0.0)
    
    return ema


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    """
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.nan_to_num(rsi_series.values, nan=50.0)
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    """
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    tr_series = pd.Series(tr)
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    atr = np.nan_to_num(atr_series.values, nan=0.0)
    
    return atr


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index using only past data.
    ADX measures trend strength (not direction).
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        
        plus_dm[i] = max(high[i] - high[i-1], 0.0)
        minus_dm[i] = max(low[i-1] - low[i], 0.0)
        
        # True DM rules
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0.0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0.0
    
    # Smooth with EMA
    tr_series = pd.Series(tr)
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    plus_di_series = (plus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean() / 
                      atr_series * 100)
    minus_di_series = (minus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean() / 
                       atr_series * 100)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di_series - minus_di_series) / (plus_di_series + minus_di_series).replace(0, np.inf)
    adx_series = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    
    adx = np.nan_to_num(adx_series.values, nan=0.0)
    
    return adx


def calculate_volume_percentile(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume percentile rank using rolling window.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_pct = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_pct
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        current_vol = volume[i]
        rank = (window < current_vol).sum() / lookback
        volume_pct[i] = rank
    
    return volume_pct


def detect_rsi_divergence(close: np.ndarray, rsi: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Detect RSI divergence using only past data.
    Returns: 1 for bullish divergence, -1 for bearish, 0 for none
    
    Bullish: Price makes lower low, RSI makes higher low
    Bearish: Price makes higher high, RSI makes lower high
    """
    n = len(close)
    divergence = np.zeros(n, dtype=np.float64)
    
    if n < lookback * 2:
        return divergence
    
    for i in range(lookback * 2, n):
        # Check for bullish divergence
        price_low_1 = np.min(close[i-lookback*2:i-lookback])
        price_low_2 = np.min(close[i-lookback:i])
        rsi_low_1 = np.min(rsi[i-lookback*2:i-lookback])
        rsi_low_2 = np.min(rsi[i-lookback:i])
        
        if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
            divergence[i] = 1.0  # Bullish
            continue
        
        # Check for bearish divergence
        price_high_1 = np.max(close[i-lookback*2:i-lookback])
        price_high_2 = np.max(close[i-lookback:i])
        rsi_high_1 = np.max(rsi[i-lookback*2:i-lookback])
        rsi_high_2 = np.max(rsi[i-lookback:i])
        
        if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
            divergence[i] = -1.0  # Bearish
    
    return divergence


def calculate_trend_strength(close: float, ema_fast: float, ema_medium: float, 
                             ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend strength score based on EMA stack alignment.
    Returns value in [-1, 1] where magnitude indicates strength.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate normalized deviations
    fast_dev = (ema_fast - ema_medium) / close
    medium_dev = (ema_medium - ema_slow) / close
    slow_dev = (ema_slow - ema_major) / close
    major_dev = (close - ema_major) / close
    
    # Direction from major trend
    major_direction = np.sign(major_dev)
    
    # Alignment score: all EMAs should be in same order
    if major_direction > 0:
        # Bullish alignment: fast > medium > slow > major
        alignment = 0.0
        if ema_fast > ema_medium:
            alignment += 0.4
        if ema_medium > ema_slow:
            alignment += 0.35
        if ema_slow > ema_major:
            alignment += 0.25
        trend_strength = alignment * major_direction
    else:
        # Bearish alignment: fast < medium < slow < major
        alignment = 0.0
        if ema_fast < ema_medium:
            alignment += 0.4
        if ema_medium < ema_slow:
            alignment += 0.35
        if ema_slow < ema_major:
            alignment += 0.25
        trend_strength = -alignment
    
    # Scale by magnitude of deviations
    avg_dev = abs(fast_dev + medium_dev + slow_dev) / 3
    trend_strength *= np.clip(avg_dev * 100, 0.5, 2.0)
    
    return np.clip(trend_strength, -1.0, 1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V1 Strategy.
    
    Signal Logic:
    1. Detect market regime using ADX (trending vs ranging)
    2. Apply different logic based on regime:
       - Trending: Follow EMA trend with momentum confirmation
       - Ranging: Mean reversion with RSI extremes
    3. RSI divergence detection for early reversal signals
    4. Volume confirmation for breakouts
    5. Volatility-adaptive position sizing
    6. Signal smoothing with hysteresis
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure valid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    volume_pct = calculate_volume_percentile(volume, VOLUME_LOOKBACK)
    rsi_divergence = detect_rsi_divergence(close, rsi, lookback=5)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK
    )
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check volatility regime
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine market regime
        is_trending = adx[i] >= ADX_TREND_THRESHOLD
        is_strong_trend = adx[i] >= ADX_STRONG_THRESHOLD
        
        # Calculate trend strength
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD
        
        # Initialize raw signal
        raw_signal = 0.0
        
        if is_trending:
            # TRENDING REGIME: Follow the trend
            min_signal_threshold = MIN_SIGNAL_TRENDING
            
            # Base signal from trend strength
            raw_signal = trend_strength
            
            # Amplify in strong trends
            if is_strong_trend:
                raw_signal *= 1.3
            
            # Momentum confirmation from RSI
            if trend_strength > 0:
                # Long bias: RSI should be bullish but not overbought
                if rsi[i] < RSI_EXTREME_HIGH:
                    rsi_factor = 1.0
                else:
                    rsi_factor = 0.5  # Reduce confidence when overbought
            else:
                # Short bias: RSI should be bearish but not oversold
                if rsi[i] > RSI_EXTREME_LOW:
                    rsi_factor = 1.0
                else:
                    rsi_factor = 0.5  # Reduce confidence when oversold
            
            raw_signal *= rsi_factor
            
            # Volume boost for trend confirmation
            if volume_confirmed:
                raw_signal *= 1.1
            
            # RSI divergence override (reversal signal)
            if rsi_divergence[i] != 0:
                # Divergence suggests potential reversal, reduce trend signal
                raw_signal *= 0.5
        
        else:
            # RANGING REGIME: Mean reversion
            min_signal_threshold = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion signal
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.5  # Long signal
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.5  # Short signal
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = 0.7
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = -0.7
            
            # RSI divergence confirmation
            if rsi_divergence[i] == 1.0:  # Bullish divergence
                raw_signal = max(raw_signal, 0.5)
            elif rsi_divergence[i] == -1.0:  # Bearish divergence
                raw_signal = min(raw_signal, -0.5)
            
            # Reduce signal if trend is fighting mean reversion
            if abs(trend_strength) > 0.3:
                raw_signal *= 0.6
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        raw_signal *= vol_factor
        
        # Apply exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply hysteresis to reduce flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            # Check if signal change exceeds hysteresis threshold
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal  # Keep previous direction
        
        # Apply minimum signal threshold
        if abs(smoothed_signal) < min_signal_threshold:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals
```

## Last Updated
2026-03-20 20:38
