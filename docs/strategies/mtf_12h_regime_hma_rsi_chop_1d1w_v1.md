# Strategy: mtf_12h_regime_hma_rsi_chop_1d1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.135 | +26.7% | -11.9% | 252 | PASS |
| ETHUSDT | 0.034 | +18.4% | -24.9% | 248 | PASS |
| SOLUSDT | 0.827 | +149.5% | -30.5% | 261 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.446 | -13.4% | -17.4% | 86 | FAIL |
| ETHUSDT | -0.229 | -1.1% | -19.9% | 86 | FAIL |
| SOLUSDT | 0.440 | +16.0% | -16.4% | 87 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #562: 12h Primary + 1d/1w HTF — Dual Regime (Choppiness) + HMA + RSI + Donchian

Hypothesis: 12h timeframe with regime-adaptive logic beats single-strategy approaches.
- When CHOP(14) > 55: Market is choppy → use mean reversion (RSI extremes)
- When CHOP(14) < 45: Market is trending → use trend following (HMA + Donchian breakout)
- 1d HMA for major trend bias (prevents counter-trend trades)
- 1w HMA for secular trend filter (only trade with long-term trend)

Why this should work:
1. 12h TF = 20-50 trades/year (optimal fee/trade ratio per Rule 10)
2. Regime detection avoids whipsaws in choppy markets (proven on ETH Sharpe +0.923)
3. HTF bias (1d/1w) prevents counter-trend trades in strong moves
4. Relaxed thresholds ensure sufficient trade generation (>=30 train, >=3 test)
5. Conservative sizing (0.28 long, 0.25 short) controls drawdown during 2022 crash

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_hma_rsi_chop_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures whether market is trending or chopping.
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use relaxed thresholds: >55 = chop, <45 = trend
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA).
    Faster and smoother than EMA, reduces lag.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # Handle NaN from rolling
    diff = 2 * wma_half - wma_full
    
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    return rsi

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

def calculate_donchian(high, low, period=14):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    hma_12h = calculate_hma(close, period=21)
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donch_upper_12h, donch_lower_12h = calculate_donchian(high, low, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.28
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
        if np.isnan(chop_12h[i]) or np.isnan(hma_12h[i]) or np.isnan(rsi_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donch_upper_12h[i]) or np.isnan(donch_lower_12h[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 55.0
        is_trending = chop_12h[i] < 45.0
        # Neutral zone: 45-55 (use trend logic as default)
        
        # === HTF TREND BIAS ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        price_above_hma = close[i] > hma_12h[i]
        price_below_hma = close[i] < hma_12h[i]
        hma_slope_up = hma_12h[i] > hma_12h[i - 5] if i >= 5 else False
        hma_slope_down = hma_12h[i] < hma_12h[i - 5] if i >= 5 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper_12h[i - 1] if i >= 1 else False
        donchian_breakout_short = close[i] < donch_lower_12h[i - 1] if i >= 1 else False
        
        # === RSI SIGNALS (relaxed for trade generation) ===
        rsi_oversold = rsi_12h[i] < 40.0
        rsi_overbought = rsi_12h[i] > 60.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: RSI oversold + HTF 1w not strongly bearish
            if rsi_oversold and not htf_1w_bearish:
                desired_signal = SIZE_LONG
            # Short: RSI overbought + HTF 1w not strongly bullish
            elif rsi_overbought and not htf_1w_bullish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Following) ===
        elif is_trending:
            # Long: HTF 1d bullish + price above HMA + (breakout OR slope up)
            if htf_1d_bullish and price_above_hma:
                if donchian_breakout_long or hma_slope_up:
                    desired_signal = SIZE_LONG
            # Short: HTF 1d bearish + price below HMA + (breakout OR slope down)
            elif htf_1d_bearish and price_below_hma:
                if donchian_breakout_short or hma_slope_down:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL (Default to trend logic with relaxed filters) ===
        else:
            # Long: HTF 1d bullish + price above HMA
            if htf_1d_bullish and price_above_hma:
                desired_signal = SIZE_LONG
            # Short: HTF 1d bearish + price below HMA
            elif htf_1d_bearish and price_below_hma:
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
                # Hold long if HTF 1d still bullish OR price above HMA
                if htf_1d_bullish or price_above_hma:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF 1d still bearish OR price below HMA
                if htf_1d_bearish or price_below_hma:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
2026-03-23 11:33
