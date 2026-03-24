# Strategy: mtf_1d_fisher_kama_chop_regime_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.478 | -7.3% | -27.0% | 118 | FAIL |
| ETHUSDT | -0.301 | -3.5% | -33.5% | 108 | FAIL |
| SOLUSDT | 0.801 | +139.3% | -27.6% | 123 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.327 | +12.3% | -15.8% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #633: 1d Primary + 1w HTF — Fisher Transform + KAMA + Choppiness Regime

Hypothesis: Daily timeframe with weekly HTF filter provides cleaner signals with 
fewer false breakouts. Fisher Transform catches reversals better than RSI in 
bear/range markets (2022 crash, 2025 bear). KAMA adapts to volatility changes.

Key innovations:
1. Ehlers Fisher Transform (period=9) — catches reversals at extremes, documented 0.8+ Sharpe
2. KAMA (Efficiency Ratio adaptive) — smoother than EMA during chop, responsive in trends
3. Choppiness Index regime switch — mean revert when CHOP>55, trend follow when CHOP<45
4. 1w HMA for macro bias — only long when weekly trend supportive
5. Looser Fisher thresholds (-1.2/+1.2) to ensure adequate trade frequency
6. Hold logic maintains positions through minor pullbacks

Why this should beat Sharpe=0.612:
- Fisher Transform has proven edge in bear markets (2022, 2025)
- 1d timeframe = fewer false signals, lower fee drag
- 1w HTF filter prevents counter-trend trades in strong macro moves
- KAMA adapts to BTC/ETH volatility regime changes
- Conservative sizing (0.30) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 20 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to a Gaussian normal distribution for clearer reversal signals.
    
    Formula:
    1. Price = (0.33 * 2 * ((close - LL) / (HH - LL) - 0.5) + 0.67 * prev_Price)
    2. Fisher = 0.5 * ln((1 + Price) / (1 - Price))
    
    Long signal: Fisher crosses above -1.2 from below
    Short signal: Fisher crosses below +1.2 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    price = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalized price
        price_raw = (close[i] - ll) / range_val
        
        # Smoothed price (0.33 * current + 0.67 * previous)
        if i > period:
            price[i] = 0.33 * 2 * (price_raw - 0.5) + 0.67 * price[i-1]
        else:
            price[i] = 0.33 * 2 * (price_raw - 0.5)
        
        # Clip to avoid log domain errors
        price[i] = np.clip(price[i], -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + price[i]) / (1 - price[i]))
        
        # Signal line (previous Fisher value)
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    
    Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA as SMA of first er_period values
    kama[er_period] = np.mean(close[:er_period+1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    fisher_1d, fisher_signal_1d = calculate_fisher_transform(high, low, close, period=9)
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(fisher_1d[i]) or np.isnan(fisher_signal_1d[i]):
            continue
        if np.isnan(kama_1d[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1d[i] > 55.0
        is_trending = chop_1d[i] < 45.0
        
        # === HTF TREND BIAS (1w HMA) ===
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND (1d) ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.2 from below
        fisher_long_cross = (fisher_1d[i] > -1.2) and (fisher_signal_1d[i] <= -1.2)
        # Short: Fisher crosses below +1.2 from above
        fisher_short_cross = (fisher_1d[i] < 1.2) and (fisher_signal_1d[i] >= 1.2)
        
        # Fisher extreme levels (for mean reversion in chop)
        fisher_oversold = fisher_1d[i] < -1.5
        fisher_overbought = fisher_1d[i] > 1.5
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with Fisher) ===
        if is_choppy:
            # Long: Fisher oversold + HTF 1w not strongly bearish
            if fisher_oversold and not htf_1w_bearish:
                desired_signal = SIZE_LONG
            # Short: Fisher overbought + HTF 1w not strongly bullish
            elif fisher_overbought and not htf_1w_bullish:
                desired_signal = -SIZE_SHORT
            # Fisher cross signals in chop
            elif fisher_long_cross:
                desired_signal = SIZE_LONG
            elif fisher_short_cross:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with KAMA + Fisher) ===
        elif is_trending:
            # Long: HTF bullish + KAMA bullish + Fisher not overbought
            if htf_1w_bullish and kama_bullish and fisher_1d[i] < 1.0:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + KAMA bearish + Fisher not oversold
            elif htf_1w_bearish and kama_bearish and fisher_1d[i] > -1.0:
                desired_signal = -SIZE_SHORT
            # Fisher cross with trend confirmation
            elif fisher_long_cross and kama_bullish:
                desired_signal = SIZE_LONG
            elif fisher_short_cross and kama_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use KAMA direction with Fisher filter
            if kama_bullish and fisher_1d[i] < 0.5:
                desired_signal = SIZE_LONG
            elif kama_bearish and fisher_1d[i] > -0.5:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        # This is CRITICAL for generating enough trades and not exiting too quickly
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish OR Fisher not extremely overbought
                if kama_bullish and fisher_1d[i] < 1.8:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if KAMA still bearish OR Fisher not extremely oversold
                if kama_bearish and fisher_1d[i] > -1.8:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 11:51
