# Strategy: keltner_squeeze_breakout_4h_trend_1h_v1

## Status
ACTIVE - Sharpe=0.168 | Return=+50.0% | DD=-18.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.270 | +13.5% | -10.7% | 28 |
| ETHUSDT | -0.276 | +9.3% | -14.7% | 69 |
| SOLUSDT | 1.051 | +127.2% | -29.0% | 7 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -2.154 | -4.5% | -6.6% | 56 |
| ETHUSDT | -1.845 | -9.0% | -9.6% | 53 |
| SOLUSDT | -1.026 | -4.2% | -10.0% | 42 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #032 - Keltner Squeeze Breakout with 4h Trend Filter (1h Primary)
==================================================================================================
Hypothesis: Current best uses 4h primary with daily filter. This uses 1h primary with 4h trend filter
for more trades while maintaining quality. Keltner Channel squeeze detection identifies low-volatility
consolidation periods, then RSI momentum triggers entries on breakout. 4h Supertrend filters counter-trend.

Key innovations vs current best (hma_rsi_pullback_daily_trend_4h_v1, Sharpe=0.537):
1. 1h PRIMARY + 4h HTF: More trade opportunities than 4h+1d, cleaner than 15m/30m
2. Keltner+Bollinger squeeze: Detects volatility compression before breakouts (proven in literature)
3. RSI momentum confirmation: RSI crossing 50 with momentum (not just pullback levels)
4. 4h Supertrend trend filter: Proven in experiments #020, #023, #030 (Sharpe 0.145-0.421)
5. Dynamic position sizing: Base size adjusted by ATR volatility (lower size in high vol)

Why this should beat Sharpe=0.537:
- Squeeze breakouts have higher win rate than pullback entries in trending markets
- 1h timeframe captures more moves than 4h while avoiding 15m/30m noise
- 4h Supertrend is stronger trend filter than daily HMA for crypto perpetuals
- Dynamic sizing reduces exposure during high volatility periods (major drawdown driver)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "keltner_squeeze_breakout_4h_trend_1h_v1"
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


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_keltner_channels(high, low, close, atr, ema_period=20, atr_mult=1.5):
    """
    Keltner Channels - EMA centerline with ATR-based bands
    Used for squeeze detection when price inside both KC and BB
    """
    n = len(close)
    if n < ema_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # EMA centerline
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return upper, ema, lower


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Bollinger Bands - SMA centerline with standard deviation bands
    Used with Keltner for squeeze detection
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < len(atr) or len(atr) == 0:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if atr[i] == 0:
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    first_valid = np.where(atr > 0)[0]
    if len(first_valid) == 0:
        return supertrend, trend
    
    start_idx = first_valid[0]
    supertrend[start_idx] = upper_band[start_idx]
    trend[start_idx] = 1
    
    for i in range(start_idx + 1, n):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            trend[i] = trend[i - 1]
            continue
        
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def calculate_squeeze_signal(bb_upper, bb_lower, kc_upper, kc_lower):
    """
    Detect volatility squeeze: Bollinger Bands inside Keltner Channels
    Returns: 1 = squeeze (low vol), 0 = normal, -1 = expansion (high vol)
    """
    n = len(bb_upper)
    squeeze = np.zeros(n)
    
    for i in range(n):
        if bb_upper[i] == 0 or kc_upper[i] == 0:
            continue
        
        # Squeeze: BB inside KC (low volatility compression)
        if bb_upper[i] <= kc_upper[i] and bb_lower[i] >= kc_lower[i]:
            squeeze[i] = 1
        # Expansion: BB outside KC (high volatility breakout)
        elif bb_upper[i] > kc_upper[i] or bb_lower[i] < kc_lower[i]:
            squeeze[i] = -1
    
    return squeeze


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    
    # Keltner Channels
    kc_upper_1h, kc_mid_1h, kc_lower_1h = calculate_keltner_channels(
        high, low, close, atr_1h, ema_period=20, atr_mult=1.5
    )
    
    # Bollinger Bands
    bb_upper_1h, bb_mid_1h, bb_lower_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Squeeze detection
    squeeze_1h = calculate_squeeze_signal(bb_upper_1h, bb_lower_1h, kc_upper_1h, kc_lower_1h)
    
    # Supertrend for stoploss reference
    _, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h ATR and Supertrend for trend direction
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # 4h RSI for momentum confirmation
        rsi_4h = calculate_rsi(close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        st_trend_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE with dynamic adjustment
    SIZE_BASE = 0.25   # Base position (25% of capital)
    SIZE_HIGH = 0.35   # High conviction (35% of capital)
    SIZE_MAX = 0.40    # Absolute maximum
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI momentum thresholds
    RSI_LONG_THRESHOLD = 55  # RSI crossing above 55 = bullish momentum
    RSI_SHORT_THRESHOLD = 45  # RSI crossing below 45 = bearish momentum
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Track squeeze state for breakout confirmation
    prev_squeeze = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            prev_squeeze[i] = prev_squeeze[i - 1] if i > 0 else 0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        st_trend_val = st_trend_1h[i]
        squeeze_val = squeeze_1h[i]
        
        # 4h trend filters (MASTER FILTER)
        st_trend_4h_val = st_trend_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        
        # Track squeeze state
        prev_squeeze[i] = prev_squeeze[i - 1] if i > 0 else 0
        
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
            
            # Stoploss check (2.0*ATR)
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
            prev_squeeze[i] = squeeze_val
            continue
        
        # ========== ENTRY LOGIC - SQUEEZE BREAKOUT WITH MOMENTUM ==========
        # Detect squeeze ending (breakout from compression)
        squeeze_ending_long = (prev_squeeze[i - 1] == 1 and squeeze_val == -1 and price > kc_mid_1h[i])
        squeeze_ending_short = (prev_squeeze[i - 1] == 1 and squeeze_val == -1 and price < kc_mid_1h[i])
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi_val > RSI_LONG_THRESHOLD and rsi_4h_val > 50
        rsi_momentum_short = rsi_val < RSI_SHORT_THRESHOLD and rsi_4h_val < 50
        
        # 4h Supertrend trend filter
        trend_filter_long = st_trend_4h_val == 1
        trend_filter_short = st_trend_4h_val == -1
        
        # LONG: Squeeze breakout + RSI momentum + 4h uptrend
        long_condition = (
            squeeze_ending_long and
            rsi_momentum_long and
            trend_filter_long and
            st_trend_val == 1
        )
        
        # SHORT: Squeeze breakout + RSI momentum + 4h downtrend
        short_condition = (
            squeeze_ending_short and
            rsi_momentum_short and
            trend_filter_short and
            st_trend_val == -1
        )
        
        # Determine position size based on conviction
        # High conviction: strong 4h trend + strong RSI momentum
        high_conviction_long = long_condition and rsi_4h_val > 60 and st_trend_4h_val == 1
        high_conviction_short = short_condition and rsi_4h_val < 40 and st_trend_4h_val == -1
        
        # Dynamic sizing: reduce size in high volatility (high ATR)
        vol_adjustment = 1.0
        if atr_4h_val > 0:
            # Normalize ATR to price for volatility percentage
            atr_pct = (atr_4h_val / price) * 100
            # Target 2% ATR, reduce size if higher
            if atr_pct > 3.0:
                vol_adjustment = 0.75
            elif atr_pct > 5.0:
                vol_adjustment = 0.50
        
        if long_condition:
            base_size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            size = min(base_size * vol_adjustment, SIZE_MAX)
            size = round(size * 4) / 4  # Discretize to 0.25 increments
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            base_size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            size = min(base_size * vol_adjustment, SIZE_MAX)
            size = round(size * 4) / 4  # Discretize to 0.25 increments
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        prev_squeeze[i] = squeeze_val
    
    return signals
```

## Last Updated
2026-03-21 18:47
