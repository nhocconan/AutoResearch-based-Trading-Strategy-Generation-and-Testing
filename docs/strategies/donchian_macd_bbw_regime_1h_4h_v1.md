# Strategy: donchian_macd_bbw_regime_1h_4h_v1

## Status
ACTIVE - Sharpe=0.025 | Return=+43.0% | DD=-22.5%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.554 | -0.5% | -19.7% | 93 |
| ETHUSDT | -0.429 | -1.3% | -18.8% | 59 |
| SOLUSDT | 1.058 | +130.8% | -29.0% | 8 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.795 | -0.6% | -8.9% | 64 |
| ETHUSDT | -0.120 | +3.6% | -8.3% | 37 |
| SOLUSDT | -0.647 | -4.4% | -20.2% | 54 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #010 - Donchian Trend + MACD Momentum + BBW Regime (1h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h+1d with HMA+RSI pullback. This uses 1h+4h with
Donchian channels for trend, MACD histogram for momentum entry, and BBW for volatility regime.

Key innovations:
1. 1h PRIMARY + 4h HTF: More trade opportunities than 4h primary, less noise than 30m
2. Donchian channels: Clean trend definition (price vs 20-period high/low), proven in #002
3. MACD histogram momentum: Entry on histogram turning positive/negative in trend direction
4. BBW regime filter: Only trade when volatility is in normal range (avoid squeeze/expansion extremes)
5. ADX confirmation: Ensure trend has strength before entering

Why this should beat #005 (Sharpe=0.537):
- 1h timeframe captures more moves than 4h while avoiding 30m noise
- Donchian trend is cleaner than HMA for breakout markets
- MACD histogram provides earlier entry signals than RSI pullback
- BBW + ADX double filter reduces false signals in choppy markets
- Different signal combo than all previous experiments (no RSI pullback)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_macd_bbw_regime_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_donchian(high, low, period=20):
    """
    Donchian Channels - trend following using highest high / lowest low
    Returns: upper_channel, lower_channel, middle_channel
    """
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2
    
    return upper, lower, middle


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    MACD Indicator
    Returns: macd_line, signal_line, histogram
    """
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_line = (ema_fast - ema_slow).values
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = strong trend, ADX < 20 = weak/choppy
    """
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * np.sum(plus_dm[i - period + 1:i + 1]) / (period * atr[i])
            minus_di[i] = 100 * np.sum(minus_dm[i - period + 1:i + 1]) / (period * atr[i])
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Bollinger Bands
    Returns: upper, middle, lower, bandwidth
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    middle = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle
    
    return upper, middle, lower, bandwidth


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Donchian channels for trend
    donch_upper_1h, donch_lower_1h, donch_mid_1h = calculate_donchian(high, low, period=20)
    
    # MACD for momentum entry
    macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # ADX for trend strength
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # Bollinger Bands for volatility regime
    bb_upper_1h, bb_mid_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h Donchian for master trend
        donch_upper_4h, donch_lower_4h, donch_mid_4h = calculate_donchian(high_4h, low_4h, period=20)
        
        # 4h ADX for trend strength confirmation
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        donch_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
    except Exception:
        donch_mid_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_BASE = 0.25   # Base position (25% of capital)
    SIZE_HIGH = 0.35   # High conviction (35% of capital)
    SIZE_MAX = 0.40    # Absolute maximum
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    # ADX thresholds
    ADX_STRONG = 25    # Strong trend
    ADX_WEAK = 20      # Weak/choppy
    
    # BBW regime thresholds (percentile-based)
    # Calculate rolling BBW percentile
    bbw_percentile = pd.Series(bbw_1h).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x, 50) if len(x) >= 50 else np.nan
    ).values
    
    first_valid = max(100, 60)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        adx_val = adx_1h[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # Donchian trend signals
        donch_mid = donch_mid_1h[i]
        donch_mid_4h = donch_mid_4h_aligned[i]
        
        # MACD momentum
        macd_hist = macd_hist_1h[i]
        macd_hist_prev = macd_hist_1h[i - 1] if i > 0 else 0
        
        # BBW regime
        bbw = bbw_1h[i]
        bbw_median = bbw_percentile[i] if not np.isnan(bbw_percentile[i]) else 0.05
        
        # ========== CHECK EXISTING POSITIONS ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== REGIME FILTER ==========
        # Only trade when ADX shows trend strength and BBW is in normal range
        regime_ok = (
            adx_val >= ADX_WEAK and  # At least some trend strength
            adx_4h_val >= ADX_WEAK and  # 4h also shows trend
            bbw > bbw_median * 0.5 and bbw < bbw_median * 2.0  # Not extreme volatility
        )
        
        if not regime_ok:
            signals[i] = 0.0
            continue
        
        # ========== TREND DIRECTION (4h Donchian Master Filter) ==========
        # Price above 4h Donchian middle = uptrend, below = downtrend
        trend_4h = 0
        if donch_mid_4h > 0 and price > donch_mid_4h:
            trend_4h = 1
        elif donch_mid_4h > 0 and price < donch_mid_4h:
            trend_4h = -1
        
        # ========== ENTRY LOGIC - MACD MOMENTUM IN TREND DIRECTION ==========
        # LONG: 4h trend up + MACD histogram turning positive + price above 1h Donchian mid
        macd_turning_long = macd_hist > 0 and macd_hist_prev <= 0
        long_condition = (
            trend_4h == 1 and
            macd_turning_long and
            price > donch_mid and
            adx_val >= ADX_STRONG  # Strong trend confirmation
        )
        
        # SHORT: 4h trend down + MACD histogram turning negative + price below 1h Donchian mid
        macd_turning_short = macd_hist < 0 and macd_hist_prev >= 0
        short_condition = (
            trend_4h == -1 and
            macd_turning_short and
            price < donch_mid and
            adx_val >= ADX_STRONG  # Strong trend confirmation
        )
        
        # Determine position size based on conviction
        # High conviction: ADX very strong + 4h ADX also strong
        high_conviction_long = long_condition and adx_val >= 35 and adx_4h_val >= 30
        high_conviction_short = short_condition and adx_val >= 35 and adx_4h_val >= 30
        
        if long_condition:
            size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            size = min(size, SIZE_MAX)
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            size = min(size, SIZE_MAX)
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals
```

## Last Updated
2026-03-21 18:22
