# Strategy: mtf_1d_kama_adx_volume_1w_bias_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.363 | -5.1% | -29.3% | 100 | FAIL |
| ETHUSDT | -0.126 | +4.7% | -24.6% | 100 | FAIL |
| SOLUSDT | 0.430 | +67.9% | -35.3% | 126 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.948 | +20.6% | -7.5% | 153 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #013: 1d KAMA-ADX Trend with 1w Bias and Volume Confirmation

Hypothesis: Based on experiment history, KAMA+ADX+Volume (#004) was the ONLY
strategy with positive Sharpe (0.514). Choppiness Index and Connors RSI all
failed with negative Sharpe. This strategy adapts the proven #004 pattern
to 1d timeframe with 1w trend bias.

Key components:
1. KAMA(10,2,30) - Adaptive MA that speeds up in trends, slows in chop
2. ADX(14) > 20 - Only trade when trend strength is sufficient
3. 1w KAMA(21) - Major trend bias (long only above, short only below)
4. Volume > 1.2x MA(20) - Confirm breakouts with volume
5. ATR(14) trailing stop at 2.5x - Protect capital

Why 1d works:
- 20-50 trades/year target (low fee drag ~1-2.5%)
- Less noise than lower TFs
- Captures multi-week swings
- Proven in research notes for bear/range markets

Position sizing: 0.25-0.30 discrete levels (CRITICAL for drawdown control)
Timeframe: 1d (REQUIRED for this experiment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_volume_1w_bias_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market noise.
    Reference: Perry Kaufman, "Trading Systems and Methods"
    
    ER = |Close - Close(n)| / Sum(|Close - Close(prev)|)
    SC = [ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)]^2
    KAMA = KAMA(prev) + SC * (Close - KAMA(prev))
    """
    close_s = pd.Series(close)
    n = len(close)
    
    kama = np.full(n, np.nan)
    
    if n < efficiency_period:
        return kama
    
    # Change over efficiency period
    change = np.abs(close - np.roll(close, efficiency_period))
    change[:efficiency_period] = np.nan
    
    # Sum of absolute changes (volatility)
    volatility = np.zeros(n)
    for i in range(efficiency_period, n):
        volatility[i] = np.sum(np.abs(close[i-efficiency_period+1:i+1] - np.roll(close[i-efficiency_period+1:i+1], 1)))
    
    # Efficiency Ratio
    er = change / np.where(volatility > 0, volatility, 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama[efficiency_period] = close[efficiency_period]
    for i in range(efficiency_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    Reference: J. Welles Wilder, 1978
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    n = len(close)
    
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Wilder's smoothing
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx_vals = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx_vals.values

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W KAMA for major trend bias
    kama_1w_21 = calculate_kama(df_1w['close'].values, 21, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    
    # Calculate 1d indicators
    kama_1d = calculate_kama(close, 10, 2, 30)
    adx_14 = calculate_adx(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volume moving average for confirmation
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1w_aligned[i]):
            continue
        
        if np.isnan(kama_1d[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === 1W MAJOR TREND BIAS ===
        weekly_bullish = close[i] > kama_1w_aligned[i]
        weekly_bearish = close[i] < kama_1w_aligned[i]
        
        # === 1D KAMA TREND ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # === KAMA SLOPE ===
        kama_slope_long = kama_1d[i] > kama_1d[i-5] if i > 5 else False
        kama_slope_short = kama_1d[i] < kama_1d[i-5] if i > 5 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Trending market (lowered from 25 for more trades)
        adx_weak = adx_14[i] < 18    # Ranging market
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 1.20 * volume_ma20[i]  # 20% above average
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need KAMA bullish + ADX strong + weekly alignment OR volume
        long_score = 0
        if kama_bullish:
            long_score += 2  # Primary requirement
        if kama_slope_long:
            long_score += 1
        if adx_strong:
            long_score += 1
        if weekly_bullish:
            long_score += 1  # Major trend alignment
        if volume_ok:
            long_score += 0.5
        
        # Enter long if score >= 3.5 (need trend + at least one confirmation)
        if long_score >= 3.5 and kama_bullish:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need KAMA bearish + ADX strong + weekly alignment OR volume
        short_score = 0
        if kama_bearish:
            short_score += 2  # Primary requirement
        if kama_slope_short:
            short_score += 1
        if adx_strong:
            short_score += 1
        if weekly_bearish:
            short_score += 1  # Major trend alignment
        if volume_ok:
            short_score += 0.5
        
        # Enter short if score >= 3.5 (need trend + at least one confirmation)
        if short_score >= 3.5 and kama_bearish:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 35 bars (~35 days on 1d), allow weaker entry
        if bars_since_last_trade > 35 and new_signal == 0.0 and not in_position:
            if kama_bullish and weekly_bullish:
                new_signal = BASE_SIZE * 0.6  # Smaller size
            elif kama_bearish and weekly_bearish:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if KAMA turns bearish
            if position_side > 0 and kama_bearish:
                trend_reversal = True
            # Exit short if KAMA turns bullish
            if position_side < 0 and kama_bullish:
                trend_reversal = True
        
        # === ADX WEAKNESS EXIT ===
        adx_exit = False
        if in_position and position_side != 0:
            # Exit if ADX drops below 18 (trend weakening)
            if adx_14[i] < 18:
                adx_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or adx_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 20:48
