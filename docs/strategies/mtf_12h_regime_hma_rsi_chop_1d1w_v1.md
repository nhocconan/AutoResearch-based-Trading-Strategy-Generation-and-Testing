# Strategy: mtf_12h_regime_hma_rsi_chop_1d1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.639 | +4.2% | -9.9% | 196 | FAIL |
| ETHUSDT | -0.864 | -4.3% | -14.2% | 201 | FAIL |
| SOLUSDT | 0.646 | +69.9% | -11.1% | 194 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.323 | +9.1% | -3.5% | 72 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #324: 12h Primary + 1d/1w HTF — Simplified Regime-Adaptive HMA/RSI v1

Hypothesis: Previous funding rate implementation failed due to data loading complexity.
Return to proven HMA trend + RSI pullback + Choppiness regime filter, but SIMPLIFIED.

Key improvements from #312 (which failed with Sharpe=-1.234):
1. REMOVED funding rate dependency (data path issues causing silent failures)
2. SIMPLIFIED regime: CHOP > 60 = choppy (mean revert), CHOP < 50 = trending (breakout)
3. LOOSENED entry thresholds to ensure 20-50 trades/year (not 0 trades)
4. ASYMMETRIC sizing: 0.30 when HTF aligned, 0.20 otherwise
5. CLEANER stoploss: 2.5x ATR from entry, properly tracked

Regime Detection:
- Choppiness Index > 60 = choppy → RSI mean reversion at extremes
- Choppiness Index < 50 = trending → HMA crossover + Donchian breakout
- 50-60 = use previous regime (hysteresis to avoid flip-flop)

Entry Logic (SIMPLIFIED):
- Choppy: RSI(14) < 30 + price > SMA200 → long; RSI > 70 + price < SMA200 → short
- Trending: HMA(21) crossover + 1d HMA alignment + Donchian breakout
- Always require 12h HMA direction confirmation

Position sizing: 0.20 base, 0.30 when 1d HTF aligned (discrete levels only)
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.40, DD>-40%, trades>=20 train, trades>=3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_hma_rsi_chop_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to volatility"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i-er_period])
        volatility = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
    
    # Smoothing constant
    sc = ((er * (2.0/(fast_period+1) - 2.0/(slow_period+1))) + 2.0/(slow_period+1)) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period+1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=10)  # For crossovers
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis (avoid flip-flop)
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 60.0
        trending_threshold = 50.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === HMA CROSSOVER (fast vs slow) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_12h_fast[i]) and not np.isnan(hma_12h_fast[i-1]):
            if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
                # Fast crosses above slow
                if hma_12h_fast[i-1] <= hma_12h[i-1] and hma_12h_fast[i] > hma_12h[i]:
                    hma_cross_long = True
                # Fast crosses below slow
                if hma_12h_fast[i-1] >= hma_12h[i-1] and hma_12h_fast[i] < hma_12h[i]:
                    hma_cross_short = True
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 35.0  # Was 30
        rsi_overbought = rsi[i] > 65.0  # Was 70
        
        # === KAMA TREND FILTER ===
        kama_bull = not np.isnan(kama[i]) and close[i] > kama[i]
        kama_bear = not np.isnan(kama[i]) and close[i] < kama[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with RSI)
        if current_regime == 2:
            # Long: RSI oversold + above SMA200 + 12h HMA bull
            if rsi_oversold and above_sma200 and hma_bull:
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            # Short: RSI overbought + below SMA200 + 12h HMA bear
            elif rsi_overbought and below_sma200 and hma_bear:
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (breakout with HMA + HTF confirmation)
        elif current_regime == 1:
            # Long: Donchian breakout + HMA bull + 1d bull + KAMA bull
            if breakout_long and hma_bull and htf_1d_bull and kama_bull:
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: Donchian breakout + HMA bear + 1d bear + KAMA bear
            elif breakout_short and hma_bear and htf_1d_bear and kama_bear:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
            
            # HMA crossover entry (alternative)
            elif hma_cross_long and hma_bull and htf_1d_bull:
                desired_signal = SIZE_BASE
            
            elif hma_cross_short and hma_bear and htf_1d_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-24 12:49
