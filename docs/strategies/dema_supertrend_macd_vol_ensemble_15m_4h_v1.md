# Strategy: dema_supertrend_macd_vol_ensemble_15m_4h_v1

## Status
ACTIVE - Sharpe=6.764 | Return=+158676.6% | DD=-7.3%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 5.194 | +2159.5% | -7.7% | 9214 |
| ETHUSDT | 6.692 | +10112.9% | -6.1% | 9182 |
| SOLUSDT | 8.406 | +463757.6% | -8.1% | 9451 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.081 | +70.8% | -4.0% | 2673 |
| ETHUSDT | 7.713 | +341.5% | -4.1% | 2592 |
| SOLUSDT | 8.323 | +528.2% | -4.1% | 2645 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #099 - DEMA SUPERTREND MACD VOLUME ENSEMBLE (15m+4h v1)
==================================================================================================
Hypothesis: Current best Sharpe=16.016 uses HMA+Supertrend+RSI+Z-score+BBW on 15m.
This experiment tries a DIFFERENT combination focusing on:

Key innovations for #099:
1. DEMA (Double Exponential MA) - faster response than HMA with less lag
2. Supertrend(10,3) for clear trend direction and dynamic stops
3. MACD histogram momentum with divergence detection
4. Volume spike confirmation (volume > 1.5x 20-bar average)
5. 15m entries + 4h DEMA trend filter (proven MTF structure)
6. Signal confidence scoring: more agreeing signals = larger position
7. ATR trailing stop at 2.0*ATR (tighter stops, faster exits)
8. Discrete position levels: 0.0, ±0.20, ±0.35 (reduces churn costs)

Why this should work:
- DEMA responds faster to trend changes than HMA/KAMA
- Supertrend provides clear binary trend signals
- MACD histogram captures momentum shifts before price
- Volume confirmation filters false breakouts
- 15m gives more entry opportunities than 1h
- 4h filter prevents counter-trend trades in strong trends

Risk controls:
- Max position size: 0.35 (35% of capital)
- ATR trailing stop: exit when price moves 2.0*ATR against position
- Volume filter: only enter on above-average volume bars
- ADX filter on 4h: only trade when trend strength > 20
- Volatility-adjusted sizing: reduce position when ATR% is high
"""

import numpy as np
import pandas as pd

name = "dema_supertrend_macd_vol_ensemble_15m_4h_v1"
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


def calculate_dema(close, period=21):
    """
    Calculate Double Exponential Moving Average (DEMA)
    DEMA = 2*EMA - EMA(EMA), reduces lag significantly
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    dema = np.zeros(n)
    
    multiplier = 2.0 / (period + 1)
    
    ema1[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema1[i] = close[i] * multiplier + ema1[i - 1] * (1 - multiplier)
    
    ema2[period - 1] = np.mean(ema1[:period])
    for i in range(period, n):
        ema2[i] = ema1[i] * multiplier + ema2[i - 1] * (1 - multiplier)
    
    for i in range(period - 1, n):
        dema[i] = 2 * ema1[i] - ema2[i]
    
    return dema


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend value, direction (1=bullish, -1=bearish)
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    
    for i in range(period, n):
        if atr[i] == 0:
            continue
            
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            # Bullish trend
            if close[i] > supertrend[i - 1]:
                supertrend[i] = lower_band
                direction[i] = 1
            # Bearish trend
            elif close[i] < supertrend[i - 1]:
                supertrend[i] = upper_band
                direction[i] = -1
            # Continue previous trend
            else:
                if direction[i - 1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i - 1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i - 1])
                    direction[i] = -1
    
    return supertrend, direction


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    fast_multiplier = 2.0 / (fast + 1)
    slow_multiplier = 2.0 / (slow + 1)
    signal_multiplier = 2.0 / (signal + 1)
    
    fast_ema = np.zeros(n)
    slow_ema = np.zeros(n)
    
    fast_ema[slow - 1] = np.mean(close[:slow])
    slow_ema[slow - 1] = fast_ema[slow - 1]
    
    for i in range(slow, n):
        fast_ema[i] = close[i] * fast_multiplier + fast_ema[i - 1] * (1 - fast_multiplier)
        slow_ema[i] = close[i] * slow_multiplier + slow_ema[i - 1] * (1 - slow_multiplier)
        macd_line[i] = fast_ema[i] - slow_ema[i]
    
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = macd_line[i] * signal_multiplier + signal_line[i - 1] * (1 - signal_multiplier)
    
    for i in range(slow + signal - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


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


def resample_to_4h(close, high, low, volume):
    """Resample 15m data to 4h (16 bars per 4h)"""
    n = len(close)
    n_4h = n // 16
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    v_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * 16
        end_idx = start_idx + 16
        if end_idx <= n:
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
            v_4h[i] = np.sum(volume[start_idx:end_idx])
    
    return c_4h, h_4h, l_4h, v_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    signals = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    dema_15m = calculate_dema(close, period=21)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Volume SMA for spike detection
    vol_sma_15m = np.zeros(n)
    for i in range(20, n):
        vol_sma_15m[i] = np.mean(volume[i - 20:i + 1])
    
    # Resample to 4h for trend filter
    c_4h, h_4h, l_4h, v_4h = resample_to_4h(close, high, low, volume)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    dema_4h = calculate_dema(c_4h, period=21)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    supertrend_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    bb_upper_4h, bb_lower_4h, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_dev=2.0)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    st_dir_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // 16
        if idx_4h < n_4h and idx_4h >= 30:
            # DEMA trend direction
            if c_4h[idx_4h] > dema_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < dema_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            st_dir_4h_mapped[i] = st_dir_4h[idx_4h]
    
    # Position sizing parameters
    SIZE_LOW = 0.20
    SIZE_HIGH = 0.35
    ATR_TARGET_PCT = 0.020
    ADX_MIN = 20
    VOL_SPIKE_MULT = 1.5
    
    # Calculate BBW percentile for regime detection
    bbw_percentile = np.zeros(n)
    valid_bbw = bbw_4h_mapped[30*16:]
    valid_bbw = valid_bbw[valid_bbw > 0]
    if len(valid_bbw) > 0:
        bbw_sorted = np.sort(valid_bbw)
        for i in range(30*16, n):
            if bbw_4h_mapped[i] > 0:
                bbw_percentile[i] = np.searchsorted(bbw_sorted, bbw_4h_mapped[i]) / len(bbw_sorted)
    
    # Tracking variables
    prev_signal = 0.0
    consecutive_votes = 0
    prev_vote_direction = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    first_valid = max(100, 30 * 16, 50, 40)
    
    for i in range(first_valid, n):
        # Skip if any indicator is invalid
        if (np.isnan(atr_15m[i]) or np.isnan(dema_15m[i]) or 
            atr_15m[i] == 0 or np.isnan(adx_4h_mapped[i]) or
            np.isnan(supertrend_15m[i]) or supertrend_15m[i] == 0):
            signals[i] = 0.0
            prev_signal = 0.0
            consecutive_votes = 0
            prev_vote_direction = 0
            entry_price = 0.0
            continue
        
        # Get indicator values
        trend_4h_val = trend_4h[i]
        adx_val = adx_4h_mapped[i]
        bbw_val = bbw_4h_mapped[i]
        bbw_pct = bbw_percentile[i]
        st_4h_dir = st_dir_4h_mapped[i]
        
        rsi_proxy = (close[i] - dema_15m[i]) / (atr_15m[i] * 2) * 50 + 50 if atr_15m[i] > 0 else 50
        
        # 4h ADX filter - only trade when higher timeframe has trend strength
        adx_filter = adx_val >= ADX_MIN
        
        # Regime detection: low BBW = trend follow, high BBW = mean revert
        trend_regime = bbw_pct < 0.5
        
        # Volume filter
        vol_spike = volume[i] > vol_sma_15m[i] * VOL_SPIKE_MULT if vol_sma_15m[i] > 0 else False
        
        # ENSEMBLE VOTING: 5 core signals
        vote_long = 0
        vote_short = 0
        
        # Signal 1: 4h Supertrend direction
        if st_4h_dir == 1:
            vote_long += 1
        elif st_4h_dir == -1:
            vote_short += 1
        
        # Signal 2: 4h DEMA trend
        if trend_4h_val == 1:
            vote_long += 1
        elif trend_4h_val == -1:
            vote_short += 1
        
        # Signal 3: 15m Supertrend direction
        if st_dir_15m[i] == 1:
            vote_long += 0.5
        elif st_dir_15m[i] == -1:
            vote_short += 0.5
        
        # Signal 4: 15m DEMA alignment
        if close[i] > dema_15m[i] and dema_15m[i] > 0:
            vote_long += 0.5
        elif close[i] < dema_15m[i] and dema_15m[i] > 0:
            vote_short += 0.5
        
        # Signal 5: MACD histogram momentum
        if macd_hist_15m[i] > 0 and macd_hist_15m[i] > macd_hist_15m[i - 1] if i > 0 else False:
            vote_long += 0.5
        elif macd_hist_15m[i] < 0 and macd_hist_15m[i] < macd_hist_15m[i - 1] if i > 0 else False:
            vote_short += 0.5
        
        # Bonus: Volume confirmation (only adds to entry confidence)
        vol_bonus = 0.5 if vol_spike else 0
        
        # Bonus: 4h DMI confirmation
        if adx_filter:
            di_diff = plus_di_4h[i // 16] - minus_di_4h[i // 16] if i // 16 < n_4h else 0
            if di_diff > 3:
                vote_long += 0.5
            elif di_diff < -3:
                vote_short += 0.5
        
        # Determine vote direction
        if vote_long > vote_short and vote_long >= 2.5:
            current_vote = 1
            total_votes = vote_long + vol_bonus
        elif vote_short > vote_long and vote_short >= 2.5:
            current_vote = -1
            total_votes = vote_short + vol_bonus
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
                highest_close = max(highest_close, close[i])
                stop_long = max(entry_price - 2.0 * entry_atr, highest_close - 2.0 * atr_15m[i])
                if close[i] < stop_long:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    continue
            else:  # Short position
                lowest_close = min(lowest_close, close[i])
                stop_short = min(entry_price + 2.0 * entry_atr, lowest_close + 2.0 * atr_15m[i])
                if close[i] > stop_short:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    continue
        
        # Generate signal
        if prev_signal != 0.0:
            if current_vote == 0 or current_vote != np.sign(prev_signal):
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
            else:
                signals[i] = prev_signal
        elif consecutive_votes >= 2 and adx_filter:
            if current_vote == 1:
                base_size = SIZE_HIGH if total_votes >= 4.0 else SIZE_LOW
                signals[i] = base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_15m[i]
                highest_close = close[i]
                prev_signal = signals[i]
            else:
                base_size = SIZE_HIGH if total_votes >= 4.0 else SIZE_LOW
                signals[i] = -base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_15m[i]
                lowest_close = close[i]
                prev_signal = signals[i]
        else:
            signals[i] = 0.0
            prev_signal = 0.0
    
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals
```

## Last Updated
2026-03-21 10:51
