# Strategy: mtf_1d_kama_chop_rsi_regime_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.174 | +7.2% | -18.6% | 139 | FAIL |
| ETHUSDT | -0.286 | -4.7% | -37.9% | 131 | FAIL |
| SOLUSDT | 0.737 | +124.4% | -32.6% | 136 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.043 | +4.8% | -16.1% | 51 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #477: 1d Primary + 1w HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Entry

Hypothesis: Based on research showing KAMA (Kaufman Adaptive Moving Average) excels in 
transitioning between trending and ranging markets by adapting its smoothing constant 
based on market noise. Combined with Choppiness Index for regime detection and RSI for 
entry timing. Key innovations:
1. KAMA(10,2,30) - adapts smoothing based on ER (Efficiency Ratio), reduces whipsaws
2. Choppiness Index(14) - regime filter: CHOP>61.8=range (mean revert), CHOP<38.2=trend
3. RSI(14) - entry timing: oversold/overbought extremes with regime-appropriate logic
4. 1w KAMA for HTF major trend bias (simpler, more stable than daily)
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
7. Relaxed entry thresholds to ensure trade generation (avoid 0-trade failure)

Why this should work: KAMA adapts to market conditions automatically (no regime switch needed).
Choppiness Index provides clear regime boundaries. RSI entries are simpler than CRSI/Fisher 
(complex oscillators have failed in recent experiments). 1d TF naturally targets 20-50 trades/year.
1w HTF ensures we trade with major trend direction. This is DIFFERENT from failed Fisher+Donchian 
combinations - using adaptive MA instead of breakout logic.

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_rsi_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing constant based on market Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER = trending (fast smoothing), Low ER = chopping (slow smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Efficiency Ratio calculation
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        net_change = np.abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            continue
        # Adaptive smoothing constant
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama, er

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = range/chop (mean reversion regime)
    CHOP < 38.2 = trending (trend follow regime)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    kama_1d, er_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    rsi_1d = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators (1w KAMA for major trend bias)
    kama_1w_raw, _ = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_1d[i]) or np.isnan(er_1d[i]):
            continue
        if np.isnan(chop_1d[i]):
            continue
        if np.isnan(rsi_1d[i]):
            continue
        if np.isnan(kama_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop = chop_1d[i] > 61.8  # Range/mean reversion regime
        is_trend = chop_1d[i] < 38.2  # Trending regime
        # Neutral zone: 38.2 <= CHOP <= 61.8
        
        # === HTF MAJOR TREND BIAS (1w KAMA) ===
        htf_bullish = close[i] > kama_1w_aligned[i]
        htf_bearish = close[i] < kama_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) ===
        price_above_kama = close[i] > kama_1d[i]
        price_below_kama = close[i] < kama_1d[i]
        kama_slope_up = kama_1d[i] > kama_1d[i - 5] if i >= 5 else False
        kama_slope_down = kama_1d[i] < kama_1d[i - 5] if i >= 5 else False
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_1d[i] < 35.0
        rsi_overbought = rsi_1d[i] > 65.0
        rsi_extreme_oversold = rsi_1d[i] < 25.0
        rsi_extreme_overbought = rsi_1d[i] > 75.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        long_score = 0
        
        # HTF bias alignment (required for trend regime)
        if htf_bullish:
            long_score += 2
        
        # Price above KAMA (trend confirmation)
        if price_above_kama:
            long_score += 1
        
        # KAMA slope up
        if kama_slope_up:
            long_score += 1
        
        # RSI entry signal (different logic per regime)
        if is_trend:
            # In trend: RSI pullback to oversold
            if rsi_oversold:
                long_score += 2
        elif is_chop:
            # In chop: RSI extreme oversold for mean reversion
            if rsi_extreme_oversold:
                long_score += 2
        else:
            # Neutral: moderate RSI oversold
            if rsi_oversold:
                long_score += 1
        
        # Enter long if score >= 4 (relaxed from 5 to ensure trades)
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # HTF bias alignment
            if htf_bearish:
                short_score += 2
            
            # Price below KAMA
            if price_below_kama:
                short_score += 1
            
            # KAMA slope down
            if kama_slope_down:
                short_score += 1
            
            # RSI entry signal
            if is_trend:
                if rsi_overbought:
                    short_score += 2
            elif is_chop:
                if rsi_extreme_overbought:
                    short_score += 2
            else:
                if rsi_overbought:
                    short_score += 1
            
            if short_score >= 4:
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
            if position_side > 0 and price_above_kama and htf_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and price_below_kama and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
2026-03-23 11:04
