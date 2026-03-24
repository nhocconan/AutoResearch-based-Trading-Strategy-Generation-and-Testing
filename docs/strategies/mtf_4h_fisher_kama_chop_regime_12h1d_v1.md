# Strategy: mtf_4h_fisher_kama_chop_regime_12h1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.194 | +3.1% | -23.2% | 429 | FAIL |
| ETHUSDT | -0.132 | +1.8% | -27.7% | 406 | FAIL |
| SOLUSDT | 0.608 | +114.2% | -36.4% | 411 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.401 | +14.5% | -17.4% | 130 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #594: 4h Primary + 12h/1d HTF — Fisher Transform + KAMA + Choppiness Regime

Hypothesis: Fisher Transform catches reversals better than RSI in bear/range markets.
Combined with KAMA (adaptive trend) and Choppiness regime detection, this should:
1. Enter at true reversal points (Fisher extremes) rather than arbitrary RSI levels
2. Adapt to volatility changes via KAMA (less whipsaw than HMA/EMA)
3. Switch between mean-revert (chop) and trend-follow (trend) based on CHOP
4. Use 12h KAMA for intermediate trend bias, 1d HMA for secular direction
5. Conservative sizing (0.30 long, 0.25 short) to survive 2022-style crashes

Why Fisher Transform over RSI:
- Fisher normalizes price to Gaussian distribution, making extremes more meaningful
- Proven Sharpe 0.8-1.5 on BTC/ETH through 2022 crash (per research notes)
- Less prone to staying at extremes during strong trends (RSI problem)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Entry when Fisher crosses above -1.5 (long) or below +1.5 (short).
    
    Steps:
    1. Calculate price position within recent range: (2*close - HH - LL) / (HH - LL)
    2. Clamp to [-0.999, 0.999] to avoid ln(0)
    3. Apply Fisher: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA for signal line
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price position within range
    with np.errstate(divide='ignore', invalid='ignore'):
        price_pos = (2 * close - hh - ll) / (hh - ll + 1e-10)
    
    # Clamp to avoid ln(0) or ln(inf)
    price_pos = np.clip(price_pos, -0.999, 0.999)
    
    # Fisher Transform
    fisher_raw = 0.5 * np.log((1 + price_pos) / (1 - price_pos + 1e-10))
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    fisher_signal = pd.Series(fisher_raw).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    return fisher, fisher_signal

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise via Efficiency Ratio (ER).
    ER = |change| / sum(|individual changes|)
    High ER = trending (fast SC), Low ER = chopping (slow SC)
    
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio calculation
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = price_change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: >55 = chop (mean revert), <45 = trend (trend follow)
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_4h[i] > 55.0
        is_trending = chop_4h[i] < 45.0
        
        # === HTF TREND BIAS ===
        htf_12h_bullish = close[i] > kama_12h_aligned[i]
        htf_12h_bearish = close[i] < kama_12h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # KAMA slope (5 bars back)
        kama_slope_up = kama_4h[i] > kama_4h[i - 5] if i >= 5 else False
        kama_slope_down = kama_4h[i] < kama_4h[i - 5] if i >= 5 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (reversal from oversold)
        fisher_cross_up = (fisher_4h[i] > -1.5) and (fisher_4h[i - 1] <= -1.5) if i >= 1 else False
        # Short: Fisher crosses below +1.5 from above (reversal from overbought)
        fisher_cross_down = (fisher_4h[i] < 1.5) and (fisher_4h[i - 1] >= 1.5) if i >= 1 else False
        
        # Extreme levels for stronger signals
        fisher_deep_oversold = fisher_4h[i] < -2.0
        fisher_deep_overbought = fisher_4h[i] > 2.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with Fisher) ===
        if is_choppy:
            # Long: Fisher deep oversold + HTF 1d not strongly bearish
            if fisher_deep_oversold and not htf_1d_bearish:
                desired_signal = SIZE_LONG
            # Short: Fisher deep overbought + HTF 1d not strongly bullish
            elif fisher_deep_overbought and not htf_1d_bullish:
                desired_signal = -SIZE_SHORT
            # Alternative: Fisher cross signals with HTF confirmation
            elif fisher_cross_up and htf_12h_bullish:
                desired_signal = SIZE_LONG
            elif fisher_cross_down and htf_12h_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Following with Fisher for entry) ===
        elif is_trending:
            # Long: HTF bullish + price above KAMA + Fisher cross up or KAMA slope up
            if htf_12h_bullish and price_above_kama:
                if fisher_cross_up or kama_slope_up:
                    desired_signal = SIZE_LONG
            # Short: HTF bearish + price below KAMA + Fisher cross down or KAMA slope down
            elif htf_12h_bearish and price_below_kama:
                if fisher_cross_down or kama_slope_down:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL (Default to KAMA trend with Fisher timing) ===
        else:
            # Long: HTF 12h bullish + price above KAMA + Fisher not overbought
            if htf_12h_bullish and price_above_kama and fisher_4h[i] < 1.5:
                desired_signal = SIZE_LONG
            # Short: HTF 12h bearish + price below KAMA + Fisher not oversold
            elif htf_12h_bearish and price_below_kama and fisher_4h[i] > -1.5:
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
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF 12h still bullish OR price above KAMA
                if htf_12h_bullish or price_above_kama:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF 12h still bearish OR price below KAMA
                if htf_12h_bearish or price_below_kama:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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
2026-03-23 11:39
