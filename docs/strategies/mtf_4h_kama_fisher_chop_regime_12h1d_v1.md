# Strategy: mtf_4h_kama_fisher_chop_regime_12h1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.107 | +24.5% | -15.5% | 238 | PASS |
| ETHUSDT | -0.188 | -0.2% | -27.3% | 235 | FAIL |
| SOLUSDT | 0.942 | +213.8% | -32.8% | 248 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.268 | +9.2% | -6.4% | 81 | PASS |
| SOLUSDT | -0.279 | -2.8% | -25.8% | 78 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #194: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Fisher Transform + Choppiness Regime

Hypothesis: The current best (CRSI + Choppiness + Donchian) works but can be improved by:
1. Using KAMA (Kaufman Adaptive MA) instead of HMA - KAMA adapts to volatility regimes better
2. Ehlers Fisher Transform for reversals - catches bear market rallies better than RSI/CRSI
3. Asymmetric entry logic - easier to enter with HTF trend, harder against it
4. Volume confirmation on breakouts to filter false signals
5. Looser regime thresholds to ensure adequate trade frequency (avoid 0-trade failure)

Key improvements over #184:
1. KAMA adapts smoothing based on volatility ratio (ER) - smoother in trends, responsive in ranges
2. Fisher Transform normalized -1 to +1, extreme readings indicate reversals
3. Dual HTF confirmation (12h + 1d) for stronger macro bias
4. Volume spike filter on breakouts (volume > 1.5x 20-bar avg)
5. More lenient CRSI thresholds (12/88 instead of 15/85) to generate more trades

TARGET: 35-55 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/101):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on Efficiency Ratio (ER).
    ER close to 1 = strong trend (use fast SC)
    ER close to 0 = choppy/range (use slow SC)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range.
    Extreme readings indicate potential reversals.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2.0
        highest = np.max(hl2)
        lowest = np.min(hl2)
        
        # Normalize to -1 to +1
        range_val = highest - lowest
        if range_val < 1e-10:
            norm = 0.0
        else:
            norm = 2.0 * (hl2[-1] - lowest) / range_val - 1.0
        
        # Clamp to avoid division issues
        norm = np.clip(norm, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + norm) / (1.0 - norm))
        if i > period:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Using slightly adjusted thresholds for better trade frequency.
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    kama_14 = calculate_kama(close, er_period=10)
    
    # Volume MA for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h KAMA for intermediate trend
    kama_12h_raw = calculate_kama(df_12h['close'].values, er_period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate 1d KAMA for macro trend
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(vol_ma20[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # Strong bias when both HTF agree
        bullish_bias = price_above_kama_12h and price_above_kama_1d
        bearish_bias = price_below_kama_12h and price_below_kama_1d
        neutral_bias = not bullish_bias and not bearish_bias
        
        # === REGIME DETECTION ===
        # Adjusted thresholds for better trade frequency
        is_range = chop_14[i] > 50.0  # Lowered from 55 to catch more range
        is_trend = chop_14[i] < 45.0  # Raised from 40 to catch more trend
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma20[i] if vol_ma20[i] > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (Fisher Transform + RSI extremes)
            # Long: Fisher < -1.2 + RSI < 35 + bullish or neutral bias
            if fisher[i] < -1.2 and rsi_14[i] < 35:
                if bullish_bias:
                    new_signal = POSITION_SIZE_FULL
                elif neutral_bias:
                    new_signal = POSITION_SIZE_HALF
                # Skip if bearish bias (counter-trend too risky)
            
            # Short: Fisher > +1.2 + RSI > 65 + bearish or neutral bias
            elif fisher[i] > 1.2 and rsi_14[i] > 65:
                if bearish_bias:
                    new_signal = -POSITION_SIZE_FULL
                elif neutral_bias:
                    new_signal = -POSITION_SIZE_HALF
                # Skip if bullish bias
        
        elif is_trend:
            # TREND FOLLOWING MODE (KAMA crossover + volume confirmation)
            # Long: Price above KAMA(14) + volume spike + bullish bias
            if close[i] > kama_14[i] and volume_spike:
                if bullish_bias:
                    new_signal = POSITION_SIZE_FULL
                elif neutral_bias:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: Price below KAMA(14) + volume spike + bearish bias
            elif close[i] < kama_14[i] and volume_spike:
                if bearish_bias:
                    new_signal = -POSITION_SIZE_FULL
                elif neutral_bias:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 12h KAMA
                if price_above_kama_12h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 12h KAMA
                if price_below_kama_12h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 12h KAMA (trend changed)
        if in_position and position_side > 0 and price_below_kama_12h:
            new_signal = 0.0
        
        # Exit short if price crosses above 12h KAMA (trend changed)
        if in_position and position_side < 0 and price_above_kama_12h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 06:23
