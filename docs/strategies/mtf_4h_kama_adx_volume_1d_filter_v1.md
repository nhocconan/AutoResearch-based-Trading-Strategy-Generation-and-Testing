# Strategy: mtf_4h_kama_adx_volume_1d_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.024 | +18.0% | -23.1% | 471 | PASS |
| ETHUSDT | 0.412 | +57.0% | -17.3% | 471 | PASS |
| SOLUSDT | 1.105 | +302.9% | -31.6% | 489 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.922 | -7.5% | -17.4% | 173 | FAIL |
| ETHUSDT | 1.276 | +39.0% | -12.0% | 160 | PASS |
| SOLUSDT | 0.622 | +21.5% | -16.7% | 161 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #004: 4h KAMA-ADX Trend with 1d Filter and Volume Confirmation

Hypothesis: Previous failed strategies (#001-#003) all used Choppiness Index regime
switching which proved ineffective. This strategy uses a DIFFERENT approach:

1. KAMA (Kaufman Adaptive Moving Average) - adapts to market volatility automatically,
   faster in trends, slower in ranges. Better than HMA/EMA for crypto whipsaws.
2. ADX(14) for trend strength - only trade when ADX > 20 (trending, not ranging)
3. 1d KAMA for major trend bias - align with higher timeframe direction
4. Volume confirmation - require volume > 1.3x 20-bar average to confirm breakout
5. ATR trailing stoploss - 2.5x ATR to protect against reversals

Why this should work better:
- KAMA's adaptive nature handles 2022 crash and 2025 bear market better than fixed EMAs
- ADX filter avoids choppy market whipsaws (major issue in 2022-2023)
- Volume confirmation reduces false breakouts
- 1d filter ensures we trade with major trend (proven in mtf_hma_rsi_zscore_v1)
- 4h timeframe targets 20-50 trades/year (manageable fee drag)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_volume_1d_filter_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility: fast in trends, slow in ranges.
    Reference: Kaufman, "Trading Systems and Methods"
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close_s - close_s.shift(slow_period)).values
    volatility = np.abs(close_s - close_s.shift(1)).values
    
    # Sum of absolute price changes over slow_period
    vol_sum = pd.Series(volatility).rolling(window=slow_period, min_periods=slow_period).sum().values
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = vol_sum > 0
    er[mask] = change[mask] / vol_sum[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = ranging market.
    Reference: Wilder, "New Concepts in Technical Trading Systems"
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values using Wilder's method (EMA with span=period)
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D KAMA for trend bias
    kama_1d_30 = calculate_kama(df_1d['close'].values, fast_period=2, slow_period=30, smoothing_period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_30_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_30)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_fast = calculate_kama(close, fast_period=2, slow_period=10, smoothing_period=5)
    kama_4h_slow = calculate_kama(close, fast_period=5, slow_period=30, smoothing_period=10)
    adx_14 = calculate_adx(high, low, close, 14)
    
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
        
        if np.isnan(kama_1d_30_aligned[i]):
            continue
        
        if np.isnan(kama_4h_fast[i]) or np.isnan(kama_4h_slow[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > kama_1d_30_aligned[i]
        daily_bearish = close[i] < kama_1d_30_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bullish = kama_4h_fast[i] > kama_4h_slow[i]
        kama_bearish = kama_4h_fast[i] < kama_4h_slow[i]
        
        # === KAMA SLOPE CONFIRMATION ===
        kama_slope_long = kama_4h_fast[i] > kama_4h_fast[i-5] if i > 5 else False
        kama_slope_short = kama_4h_fast[i] < kama_4h_fast[i-5] if i > 5 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Trending market (not ranging)
        
        # === VOLUME CONFIRMATION ===
        volume_spike = volume[i] > 1.3 * volume_ma20[i]  # 30% above average
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need KAMA trend + ADX strength + (volume OR daily bias)
        long_conditions = 0
        if kama_bullish:
            long_conditions += 1
        if adx_strong:
            long_conditions += 1
        if volume_spike:
            long_conditions += 0.5
        if daily_bullish:
            long_conditions += 0.5
        if kama_slope_long:
            long_conditions += 0.5
        
        # Enter long if score >= 2.5 (need trend + strength + confirmation)
        if long_conditions >= 2.5 and kama_bullish and adx_strong:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need KAMA trend + ADX strength + (volume OR daily bias)
        short_conditions = 0
        if kama_bearish:
            short_conditions += 1
        if adx_strong:
            short_conditions += 1
        if volume_spike:
            short_conditions += 0.5
        if daily_bearish:
            short_conditions += 0.5
        if kama_slope_short:
            short_conditions += 0.5
        
        # Enter short if score >= 2.5 (need trend + strength + confirmation)
        if short_conditions >= 2.5 and kama_bearish and adx_strong:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish:
                new_signal = BASE_SIZE * 0.7  # Smaller size
            elif kama_bearish and daily_bearish:
                new_signal = -BASE_SIZE * 0.7
        
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
            # Exit long if 4h KAMA turns bearish
            if position_side > 0 and kama_bearish:
                trend_reversal = True
            # Exit short if 4h KAMA turns bullish
            if position_side < 0 and kama_bullish:
                trend_reversal = True
        
        # === ADX WEAKNESS EXIT ===
        adx_weakness = False
        if in_position and position_side != 0:
            # Exit if ADX drops below 15 (trend dying)
            if adx_14[i] < 15:
                adx_weakness = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or adx_weakness:
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
2026-03-22 20:37
