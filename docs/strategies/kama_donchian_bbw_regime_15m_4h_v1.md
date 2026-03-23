# Strategy: kama_donchian_bbw_regime_15m_4h_v1

## Status
ACTIVE - Sharpe=7.409 | Return=+190088.6% | DD=-7.5%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 6.109 | +3529.7% | -7.4% | 6710 |
| ETHUSDT | 7.612 | +22377.8% | -9.0% | 6655 |
| SOLUSDT | 8.506 | +544358.1% | -6.0% | 6556 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 6.372 | +120.2% | -2.8% | 1894 |
| ETHUSDT | 9.145 | +469.7% | -3.4% | 1849 |
| SOLUSDT | 9.303 | +677.1% | -4.4% | 1789 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #098 - KAMA DONCHIAN BREAKOUT WITH BBW REGIME (15m+4h v1)
==================================================================================================
Hypothesis: Current best Sharpe=16.016 uses HMA+Supertrend+RSI+Z-score+BBW on 15m.
This experiment tries a DIFFERENT combination that hasn't been heavily tested:

Key innovations for #098:
1. KAMA (Kaufman Adaptive MA) instead of HMA - adapts speed to market efficiency
2. Donchian Channel breakout (20-period) for clear entry signals
3. Bollinger Band Width regime: low BW = trend follow, high BW = mean revert
4. 15m entries + 4h KAMA trend filter (proven MTF structure from exp#090/095/096)
5. RSI filter to avoid chasing overextended breakouts
6. ATR trailing stop at 2.5*ATR (proven in exp#097)
7. Discrete position levels: 0.0, ±0.20, ±0.35 (reduces churn)
8. Entry confirmation: 2 consecutive bars with same signal direction

Why this should work:
- KAMA is more adaptive than HMA in changing volatility regimes
- Donchian breakouts capture momentum moves clearly
- BBW regime detection switches between trend/mean-revert modes
- 15m gives more entry opportunities than 1h while staying stable
- 4h filter prevents counter-trend trades

Risk controls:
- Max position size: 0.35 (35% of capital)
- ATR trailing stop: exit when price moves 2.5*ATR against position
- ADX filter on 4h: only trade when trend strength > 20
- Volatility-adjusted sizing: reduce position when ATR% is high
"""

import numpy as np
import pandas as pd

name = "kama_donchian_bbw_regime_15m_4h_v1"
timeframe = "15m"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - faster in trends, slower in chop
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_dev * std
        lower[i] = sma[i] - std_dev * std
        if sma[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bandwidth


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def resample_to_4h(close, high, low):
    """Resample 15m data to 4h (16 bars per 4h)"""
    n = len(close)
    n_4h = n // 16
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * 16
        end_idx = start_idx + 16
        if end_idx <= n:
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
    
    return c_4h, h_4h, l_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    kama_15m = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    donchian_upper_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    bb_upper_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Resample to 4h for trend filter
    c_4h, h_4h, l_4h = resample_to_4h(close, high, low)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    bb_upper_4h, bb_lower_4h, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_dev=2.0)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    di_diff_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // 16
        if idx_4h < n_4h and idx_4h >= 30:
            # KAMA trend direction
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            di_diff_4h[i] = plus_di_4h[idx_4h] - minus_di_4h[idx_4h]
    
    # Position sizing parameters
    SIZE_LOW = 0.20
    SIZE_HIGH = 0.35
    ATR_TARGET_PCT = 0.020
    ADX_MIN = 20
    
    # Calculate BBW percentile for regime detection
    bbw_percentile = np.zeros(n)
    valid_bbw = bbw_4h_mapped[30*16:]  # Skip warmup
    if len(valid_bbw) > 0:
        for i in range(30*16, n):
            if bbw_4h_mapped[i] > 0:
                bbw_percentile[i] = np.sum(valid_bbw[:len(valid_bbw)] <= bbw_4h_mapped[i]) / len(valid_bbw)
    
    # Tracking variables
    prev_signal = 0.0
    consecutive_votes = 0
    prev_vote_direction = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    first_valid = max(100, 30 * 16, 40, 34)
    
    for i in range(first_valid, n):
        # Skip if any indicator is invalid
        if (np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or 
            atr_15m[i] == 0 or np.isnan(adx_4h_mapped[i]) or
            np.isnan(kama_15m[i]) or kama_15m[i] == 0):
            signals[i] = 0.0
            prev_signal = 0.0
            consecutive_votes = 0
            prev_vote_direction = 0
            entry_price = 0.0
            continue
        
        # Get indicator values
        trend = trend_4h[i]
        adx_val = adx_4h_mapped[i]
        bbw_val = bbw_4h_mapped[i]
        bbw_pct = bbw_percentile[i]
        di_diff = di_diff_4h[i]
        rsi_val = rsi_15m[i]
        
        # 4h ADX filter - only trade when higher timeframe has trend strength
        adx_filter = adx_val >= ADX_MIN
        
        # Regime detection: low BBW = trend follow, high BBW = mean revert
        trend_regime = bbw_pct < 0.5  # Lower 50% = trend regime
        
        # ENSEMBLE VOTING: 3 core signals
        vote_long = 0
        vote_short = 0
        
        # Signal 1: 4h KAMA trend
        if trend == 1:
            vote_long += 1
        elif trend == -1:
            vote_short += 1
        
        # Signal 2: 15m Donchian breakout
        if close[i] > donchian_upper_15m[i] and donchian_upper_15m[i] > 0:
            if trend_regime:  # Trend regime - follow breakout
                vote_long += 1
            else:  # Mean revert regime - fade breakout
                vote_short += 0.5
        elif close[i] < donchian_lower_15m[i] and donchian_lower_15m[i] > 0:
            if trend_regime:  # Trend regime - follow breakout
                vote_short += 1
            else:  # Mean revert regime - fade breakout
                vote_long += 0.5
        
        # Signal 3: 15m KAMA alignment
        if close[i] > kama_15m[i] and kama_15m[i] > 0:
            vote_long += 0.5
        elif close[i] < kama_15m[i] and kama_15m[i] > 0:
            vote_short += 0.5
        
        # Signal 4: RSI filter (avoid overextended entries)
        if trend == 1 and rsi_val > 45 and rsi_val < 70:
            vote_long += 0.5
        elif trend == -1 and rsi_val < 55 and rsi_val > 30:
            vote_short += 0.5
        
        # Bonus: 4h DMI confirmation
        if adx_filter:
            if di_diff > 3:
                vote_long += 0.5
            elif di_diff < -3:
                vote_short += 0.5
        
        # Determine vote direction
        if vote_long > vote_short and vote_long >= 2.0:
            current_vote = 1
            total_votes = vote_long
        elif vote_short > vote_long and vote_short >= 2.0:
            current_vote = -1
            total_votes = vote_short
        else:
            current_vote = 0
            total_votes = 0
        
        # Hysteresis: 2 consecutive bars for entry
        if current_vote != 0 and current_vote == prev_vote_direction:
            consecutive_votes += 1
        elif current_vote != 0:
            consecutive_votes = 1
            prev_vote_direction = current_vote
        else:
            consecutive_votes = 0
            prev_vote_direction = 0
        
        # Calculate volatility-adjusted size
        atr_pct = atr_15m[i] / close[i] if close[i] > 0 else 0
        vol_adjustment = min(1.3, max(0.6, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
        
        # Check for ATR trailing stop exit
        if prev_signal != 0.0 and entry_price > 0:
            if prev_signal > 0:  # Long position
                # Update highest close for trailing
                highest_close = max(highest_close, close[i])
                # Stop loss: 2.5 * ATR below entry or highest
                stop_long = max(entry_price - 2.5 * entry_atr, highest_close - 2.5 * atr_15m[i])
                if close[i] < stop_long:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    continue
            else:  # Short position
                # Update lowest close for trailing
                lowest_close = min(lowest_close, close[i])
                # Stop loss: 2.5 * ATR above entry or lowest
                stop_short = min(entry_price + 2.5 * entry_atr, lowest_close + 2.5 * atr_15m[i])
                if close[i] > stop_short:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    continue
        
        # Generate signal
        if prev_signal != 0.0:
            # Already in position
            if current_vote == 0 or current_vote != np.sign(prev_signal):
                # Exit on vote reversal or neutral
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
            else:
                # Hold position
                signals[i] = prev_signal
        elif consecutive_votes >= 2 and adx_filter:
            # New entry
            if current_vote == 1:
                base_size = SIZE_HIGH if total_votes >= 3.5 else SIZE_LOW
                signals[i] = base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_15m[i]
                highest_close = close[i]
                prev_signal = signals[i]
            else:
                base_size = SIZE_HIGH if total_votes >= 3.5 else SIZE_LOW
                signals[i] = -base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_15m[i]
                lowest_close = close[i]
                prev_signal = signals[i]
        else:
            signals[i] = 0.0
            prev_signal = 0.0
    
    # Clip to max position size
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals
```

## Last Updated
2026-03-21 10:50
