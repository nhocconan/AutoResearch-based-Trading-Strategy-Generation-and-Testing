# Strategy: mtf_4h_kama_chop_regime_volume_12h_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.183 | +3.8% | -23.4% | 298 | FAIL |
| ETHUSDT | -0.040 | +9.3% | -30.1% | 282 | FAIL |
| SOLUSDT | 0.482 | +80.1% | -32.3% | 302 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.102 | +6.2% | -20.0% | 95 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1094: 4h Primary + 12h HTF — KAMA Adaptive Trend + Choppiness Regime + Volume

Hypothesis: After 794+ failed experiments, the key insight is:
1. KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than HMA/EMA
   - Fast in trends (ER high), slow in chop (ER low)
   - Reduces whipsaws in 2022 crash and 2025 bear market
2. Choppiness Index (CHOP) regime filter:
   - CHOP > 61.8 = range market → use mean reversion (RSI extremes)
   - CHOP < 38.2 = trending market → use trend following (KAMA direction)
   - This dual-regime approach works in BOTH bull and bear markets
3. 12h KAMA for macro trend filter (same as winning baseline)
4. Volume confirmation on breakouts (avoid fake breakouts)
5. More lenient RSI thresholds to ensure ≥10 trades/year

Why this should beat Sharpe=0.612:
- KAMA reduces whipsaws vs HMA in volatile 2022
- Choppiness filter switches between mean-reversion and trend-following modes
- Volume confirmation filters out false breakouts
- Lenient RSI ensures adequate trade frequency

Timeframe: 4h (primary)
HTF: 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_regime_volume_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    
    Formula:
    1. Efficiency Ratio (ER) = |change| / sum(|changes|) over period
       ER = 1 in strong trend, ER = 0 in chop
    2. Smoothing Constant (SC) = (ER * (fast_SC - slow_SC) + slow_SC)^2
       fast_SC = 2/(fast+1), slow_SC = 2/(slow+1)
    3. KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + 1:
        return kama
    
    # Calculate Efficiency Ratio
    change = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(n - period):
        volatility[i] = np.sum(np.abs(np.diff(close[i:i+period+1])))
    
    er = np.zeros(n)
    mask = volatility > 1e-10
    er[period:][mask] = change[mask] / volatility[mask]
    er[:period] = er[period] if period < n else 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market consolidation vs trending.
    
    Formula:
    1. ATR_sum = sum(ATR(1)) over period
    2. TR_sum = sum(True Range) over period
    3. CHOP = 100 * log10(ATR_sum / (max_high - min_low)) / log10(period)
    
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling calculations
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h KAMA for macro trend filter
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate primary (4h) indicators
    kama_10 = calculate_kama(close, period=10)
    kama_30 = calculate_kama(close, period=30)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(kama_12h_aligned[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            continue
        
        # === MACRO TREND (12h KAMA) ===
        macro_bull = close[i] > kama_12h_aligned[i]
        macro_bear = close[i] < kama_12h_aligned[i]
        
        # === PRIMARY TREND (4h KAMA crossover) ===
        kama_bull = kama_10[i] > kama_30[i]
        kama_bear = kama_10[i] < kama_30[i]
        
        # === CHOPPINESS REGIME ===
        choppy_market = chop[i] > 55.0  # Range market
        trending_market = chop[i] < 45.0  # Trend market
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = volume[i] > 1.2 * vol_sma[i]
        
        # === RSI SIGNALS ===
        # More lenient thresholds to ensure trades
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral_long = 40.0 <= rsi[i] <= 55.0
        rsi_neutral_short = 45.0 <= rsi[i] <= 60.0
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 1.8 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING MARKET (CHOP < 45) ===
        # Use trend-following logic
        if trending_market:
            # Long: macro bull + KAMA bull + volume confirmation
            if macro_bull and kama_bull and vol_above_avg:
                desired_signal = current_size
            # Short: macro bear + KAMA bear + volume confirmation
            elif macro_bear and kama_bear and vol_above_avg:
                desired_signal = -current_size
        
        # === REGIME 2: CHOPPY/RANGE MARKET (CHOP > 55) ===
        # Use mean-reversion logic
        elif choppy_market:
            # Long: RSI oversold + price above 12h KAMA (macro support)
            if rsi_oversold and macro_bull:
                desired_signal = current_size
            # Short: RSI overbought + price below 12h KAMA (macro resistance)
            elif rsi_overbought and macro_bear:
                desired_signal = -current_size
        
        # === REGIME 3: TRANSITION (45 <= CHOP <= 55) ===
        # Use neutral RSI pullback entries (like original #1084)
        else:
            if macro_bull and kama_bull and rsi_neutral_long:
                desired_signal = current_size
            elif macro_bear and kama_bear and rsi_neutral_short:
                desired_signal = -current_size
        
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
                # Hold long if KAMA still bullish
                if kama_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if KAMA still bearish
                if kama_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA crosses bearish or macro reverses strongly
            if kama_bear and rsi[i] > 70.0:
                desired_signal = 0.0
            if macro_bear and chop[i] < 40.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA crosses bullish or macro reverses strongly
            if kama_bull and rsi[i] < 30.0:
                desired_signal = 0.0
            if macro_bull and chop[i] < 40.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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
2026-03-23 19:41
