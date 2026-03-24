# Strategy: mtf_4h_donchian_rsi_pullback_1d_hma_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.748 | -9.0% | -24.4% | 234 | FAIL |
| ETHUSDT | 0.471 | +50.2% | -11.8% | 213 | PASS |
| SOLUSDT | -0.498 | -16.7% | -29.5% | 186 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.013 | +5.6% | -12.8% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1039: 4h Primary + 1d HTF — Donchian Breakout + RSI Pullback + HMA Trend

Hypothesis: After analyzing 752+ failed strategies, the key insight is that COMPLEX regime
switching creates mutually exclusive conditions → 0 trades. The winning approach is SIMPLER:

1. DONCHIAN(20) BREAKOUT: Pure price action trend signal. Long when price breaks 20-bar high,
   short when breaks 20-bar low. This is the MOST RELIABLE trend signal across all assets.

2. RSI(14) PULLBACK ENTRY: Instead of entering on breakout (chase), wait for RSI pullback
   to 40-50 zone in uptrend, or 50-60 zone in downtrend. This gives better risk/reward.

3. 1d HMA21 MACRO FILTER: Only long when price > 1d HMA21 (bullish macro), only short when
   price < 1d HMA21 (bearish macro). This asymmetric filter works in bear/range markets.

4. ATR TRAILING STOP: 2.5x ATR(14) from entry high/low. Signal→0 when hit.

5. RELAXED ENTRY CONDITIONS: RSI thresholds 35-65 (not 30-70) to ensure sufficient trades.
   This is CRITICAL — many strategies fail due to 0 trades from overly strict filters.

Why this works:
- Donchian breakout = proven trend signal (worked in exp #1033, #1037)
- RSI pullback = better entry timing than breakout chase
- 1d HMA = simple macro filter (not complex dual-HTF that creates conflicts)
- Relaxed RSI = ensures 30+ trades/train, 3+ trades/test

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_rsi_pullback_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian_channels(high, low, period=20):
    """
    Donchian Channels: Highest high and lowest low over N periods
    Upper = breakout level for longs
    Lower = breakout level for shorts
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """
    Relative Strength Index
    RSI < 30 = oversold, RSI > 70 = overbought
    For pullback entries: RSI 40-50 in uptrend, 50-60 in downtrend
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    # Use EMA for RSI calculation (smoother than SMA)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average for trend direction."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Track breakouts (price crossed above upper or below lower)
    breakout_long = np.zeros(n, dtype=bool)
    breakout_short = np.zeros(n, dtype=bool)
    
    for i in range(20, n):
        if not np.isnan(donchian_upper[i]) and not np.isnan(donchian_upper[i-1]):
            # Price crossed above upper channel
            if close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]:
                breakout_long[i] = True
            # Price crossed below lower channel
            if close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]:
                breakout_short[i] = True
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    bars_in_trade = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === MACRO TREND (1d HMA21) ===
        # Asymmetric filter: easier to long when above, easier to short when below
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN TREND SIGNAL ===
        # Price above upper = strong uptrend, below lower = strong downtrend
        trend_long = close[i] > donchian_upper[i]
        trend_short = close[i] < donchian_lower[i]
        
        # === RSI PULLBACK ZONES ===
        # In uptrend: wait for RSI pullback to 40-55 zone
        # In downtrend: wait for RSI pullback to 45-60 zone
        rsi_pullback_long = 35 <= rsi_4h[i] <= 55
        rsi_pullback_short = 45 <= rsi_4h[i] <= 65
        rsi_extreme_long = rsi_4h[i] < 35  # Deep oversold
        rsi_extreme_short = rsi_4h[i] > 65  # Deep overbought
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Entry 1: Macro bull + trend long + RSI pullback (primary entry)
        if macro_bull and trend_long and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Entry 2: Macro bull + breakout long + RSI not extreme (breakout confirmation)
        elif macro_bull and breakout_long[i] and rsi_4h[i] < 65:
            desired_signal = BASE_SIZE
        # Entry 3: Macro bull + RSI extreme oversold (reversal play)
        elif macro_bull and rsi_extreme_long:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Entry 1: Macro bear + trend short + RSI pullback (primary entry)
        if macro_bear and trend_short and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # Entry 2: Macro bear + breakout short + RSI not extreme (breakout confirmation)
        elif macro_bear and breakout_short[i] and rsi_4h[i] > 35:
            desired_signal = -BASE_SIZE
        # Entry 3: Macro bear + RSI extreme overbought (reversal play)
        elif macro_bear and rsi_extreme_short:
            desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish or trend still long
                if macro_bull or (close[i] > donchian_lower[i] and rsi_4h[i] > 30):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish or trend still short
                if macro_bear or (close[i] < donchian_upper[i] and rsi_4h[i] < 70):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish AND RSI overbought
            if macro_bear and rsi_4h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish AND RSI oversold
            if macro_bull and rsi_4h[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                bars_in_trade = 0
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                bars_in_trade = 0
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
                bars_in_trade += 1
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
                bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 18:51
