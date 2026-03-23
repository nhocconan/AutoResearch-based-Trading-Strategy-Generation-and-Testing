# Strategy: kama_donchian_adx_zscore_regime_15m_4h_v1

## Status
ACTIVE - Sharpe=5.353 | Return=+214552.0% | DD=-12.6%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.423 | +2616.5% | -11.1% | 10602 |
| ETHUSDT | 5.437 | +14659.0% | -12.3% | 10375 |
| SOLUSDT | 6.200 | +626380.4% | -14.5% | 9972 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 3.788 | +80.6% | -3.6% | 3286 |
| ETHUSDT | 5.704 | +288.2% | -6.5% | 3027 |
| SOLUSDT | 6.572 | +525.6% | -5.5% | 2917 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #058 - KAMA_DONCHIAN_ADX_ZSCORE_REGIME_15M_4H_V1
==================================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adjusts its smoothing based on market efficiency,
making it superior to HMA/EMA in both trending and ranging markets. Combining KAMA with Donchian
channel breakouts (pure price action) + ADX trend strength filter + Z-score mean reversion creates
a robust regime-adaptive system. 15m entries with 4h trend filter proven in experiments #047, #049,
#053, #054, #056 (all Sharpe > 7.0).

Key innovations:
- KAMA EFFICIENCY RATIO: Adaptive smoothing that speeds up in trends, slows in noise
- DONCHIAN BREAKOUTS: Pure price action - 20-bar high/low breaks with ADX confirmation
- Z-SCORE MEAN REVERSION: Enter counter-trend when Z-score > 2.0 in low-ADX regimes
- ADX REGIME FILTER: ADX > 25 = trend follow, ADX < 20 = mean revert
- 15M/4H MULTI-TF: Proven combination from top-performing strategies
- CONFIDENCE SIZING: Position size scales with signal agreement (3+ signals = max size)

Why this should beat #057 (Sharpe=3.043) and approach #049 (Sharpe=13.974):
- KAMA is more adaptive than HMA in changing volatility regimes
- Donchian breakouts capture pure momentum without lag
- Z-score provides mean reversion entries that HMA/Supertrend miss
- ADX regime filter prevents trend-following in choppy markets
- Based on winning multi-TF architecture from experiments #047, #049, #053, #054, #056

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (proven to control drawdown in 2022 crash)
- MIN signal: 0.20 (avoid tiny positions eaten by fees)
- Discrete levels: 0.0, 0.20, 0.28, 0.35 (reduces churn costs)
- Stoploss: 2.5*ATR trailing, TP at 2R then trail at 1R
- Regime confidence multiplier: 0.5-1.0 based on signal agreement
"""

import numpy as np
import pandas as pd

name = "kama_donchian_adx_zscore_regime_15m_4h_v1"
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
    KAMA adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth +DM, -DM, and TR
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    tr_smooth = np.zeros(n)
    
    plus_dm_smooth[period] = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period + 1])
    tr_smooth[period] = np.sum(tr[1:period + 1])
    
    for i in range(period + 1, n):
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        if sma[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bbw


def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank"""
    n = len(values)
    percentile = np.zeros(n)
    
    for i in range(window - 1, n):
        valid_vals = values[i - window + 1:i + 1]
        valid_vals = valid_vals[~np.isnan(valid_vals)]
        if len(valid_vals) > 0:
            current_val = values[i]
            percentile[i] = np.sum(valid_vals <= current_val) / len(valid_vals)
    
    return percentile


def resample_to_higher_tf(close, high, low, bars_per_tf=16):
    """Resample 15m data to 4h (16 x 15m = 4h)"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return close.copy(), high.copy(), low.copy()
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    kama_15m = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx_15m = calculate_adx(high, low, close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    donchian_upper_15m, donchian_lower_15m = calculate_donchian_channels(high, low, period=20)
    _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h = resample_to_higher_tf(close, high, low, bars_per_4h)
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian_channels(h_4h, l_4h, period=20)
    _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_percentile_4h = calculate_percentile_rank(bbw_4h, window=50)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    donchian_trend_4h = np.zeros(n)
    
    n_4h = len(c_4h)
    for i in range(n):
        idx_4h = i // bars_per_4h
        
        if idx_4h < n_4h and idx_4h >= 40:
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
            
            # Donchian trend (price position in channel)
            channel_width = donchian_upper_4h[idx_4h] - donchian_lower_4h[idx_4h]
            if channel_width > 0:
                price_position = (c_4h[idx_4h] - donchian_lower_4h[idx_4h]) / channel_width
                if price_position > 0.6:
                    donchian_trend_4h[i] = 1
                elif price_position < 0.4:
                    donchian_trend_4h[i] = -1
    
    # Position sizing parameters (DISCRETE levels)
    SIZE_LEVELS = np.array([0.0, 0.20, 0.28, 0.35])
    BASE_SIZE = 0.28
    
    # Signal thresholds
    ADX_TREND_THRESHOLD = 25
    ADX_RANGE_THRESHOLD = 20
    ZSCORE_EXTREME = 2.0
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    
    # Stoploss multipliers
    ATR_STOP = 2.5
    
    # Regime thresholds
    BBW_LOW_PERCENTILE = 0.30
    BBW_HIGH_PERCENTILE = 0.70
    
    first_valid = max(200, 40 * bars_per_4h, 50)
    
    # Generate signals with regime-switching
    signals = np.zeros(n)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Hysteresis counters
    long_confirm_count = np.zeros(n, dtype=int)
    short_confirm_count = np.zeros(n, dtype=int)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        adx_4h = adx_4h_mapped[i]
        bbw_4h_val = bbw_4h_mapped[i]
        atr = atr_15m[i]
        price = close[i]
        
        # 15m signals
        kama_15m_val = kama_15m[i]
        adx_15m_val = adx_15m[i]
        zscore_val = zscore_15m[i]
        rsi_val = rsi_15m[i]
        donchian_upper = donchian_upper_15m[i]
        donchian_lower = donchian_lower_15m[i]
        donchian_trend = donchian_trend_4h[i]
        
        # Determine regime
        is_trend_regime = adx_4h > ADX_TREND_THRESHOLD
        is_range_regime = adx_4h < ADX_RANGE_THRESHOLD
        bbw_pct = bbw_percentile_4h[i // bars_per_4h] if i // bars_per_4h < len(bbw_percentile_4h) else 0.5
        is_low_vol = bbw_pct < BBW_LOW_PERCENTILE
        is_high_vol = bbw_pct > BBW_HIGH_PERCENTILE
        
        # Calculate signal scores
        # Signal 1: 4h KAMA trend
        kama_signal = 0
        if trend == 1:
            kama_signal = 1
        elif trend == -1:
            kama_signal = -1
        
        # Signal 2: 4h Donchian position
        donchian_signal = 0
        if donchian_trend == 1:
            donchian_signal = 1
        elif donchian_trend == -1:
            donchian_signal = -1
        
        # Signal 3: 15m KAMA
        kama_15m_signal = 0
        if price > kama_15m_val:
            kama_15m_signal = 1
        elif price < kama_15m_val:
            kama_15m_signal = -1
        
        # Signal 4: Donchian breakout
        breakout_signal = 0
        if price > donchian_upper * 0.999:  # Near upper break
            breakout_signal = 1
        elif price < donchian_lower * 1.001:  # Near lower break
            breakout_signal = -1
        
        # Signal 5: Z-score mean reversion
        zscore_signal = 0
        if zscore_val < -ZSCORE_EXTREME:
            zscore_signal = 1  # Oversold, expect mean reversion up
        elif zscore_val > ZSCORE_EXTREME:
            zscore_signal = -1  # Overbought, expect mean reversion down
        
        # Signal 6: RSI
        rsi_signal = 0
        if rsi_val < RSI_LONG_MAX:
            rsi_signal = 1
        elif rsi_val > RSI_SHORT_MIN:
            rsi_signal = -1
        
        # Signal 7: 15m ADX trend strength
        adx_signal = 0
        if adx_15m_val > ADX_TREND_THRESHOLD:
            if price > kama_15m_val:
                adx_signal = 1
            elif price < kama_15m_val:
                adx_signal = -1
        
        # Calculate weighted signal score based on regime
        if is_trend_regime:
            # Trend-following regime: weight trend signals highest
            long_score = (
                0.25 * (kama_signal == 1) +
                0.20 * (donchian_signal == 1) +
                0.20 * (kama_15m_signal == 1) +
                0.15 * (breakout_signal == 1) +
                0.10 * (rsi_signal == 1) +
                0.10 * (adx_signal == 1)
            )
            short_score = (
                0.25 * (kama_signal == -1) +
                0.20 * (donchian_signal == -1) +
                0.20 * (kama_15m_signal == -1) +
                0.15 * (breakout_signal == -1) +
                0.10 * (rsi_signal == -1) +
                0.10 * (adx_signal == -1)
            )
        elif is_range_regime:
            # Mean reversion regime: weight Z-score/RSI higher
            long_score = (
                0.30 * (zscore_signal == 1) +
                0.25 * (rsi_signal == 1) +
                0.15 * (kama_15m_signal == 1) +
                0.15 * (breakout_signal == -1) +  # False breakout long
                0.10 * (kama_signal == 1) +
                0.05 * (donchian_signal == 1)
            )
            short_score = (
                0.30 * (zscore_signal == -1) +
                0.25 * (rsi_signal == -1) +
                0.15 * (kama_15m_signal == -1) +
                0.15 * (breakout_signal == 1) +  # False breakout short
                0.10 * (kama_signal == -1) +
                0.05 * (donchian_signal == -1)
            )
        else:
            # Neutral regime: balanced weights
            long_score = (
                0.20 * (kama_signal == 1) +
                0.15 * (donchian_signal == 1) +
                0.15 * (kama_15m_signal == 1) +
                0.15 * (breakout_signal == 1) +
                0.15 * (zscore_signal == 1) +
                0.10 * (rsi_signal == 1) +
                0.10 * (adx_signal == 1)
            )
            short_score = (
                0.20 * (kama_signal == -1) +
                0.15 * (donchian_signal == -1) +
                0.15 * (kama_15m_signal == -1) +
                0.15 * (breakout_signal == -1) +
                0.15 * (zscore_signal == -1) +
                0.10 * (rsi_signal == -1) +
                0.10 * (adx_signal == -1)
            )
        
        # HYSTERESIS: Update confirmation counters
        if long_score >= 0.45:
            long_confirm_count[i] = long_confirm_count[i - 1] + 1 if i > 0 else 1
        else:
            long_confirm_count[i] = 0
        
        if short_score >= 0.45:
            short_confirm_count[i] = short_confirm_count[i - 1] + 1 if i > 0 else 1
        else:
            short_confirm_count[i] = 0
        
        # Check stoploss and take profit for existing positions
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
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP * atr
                if not prev_tp and price >= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        long_confirm_count[i] = 0
                        short_confirm_count[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP * atr
                if not prev_tp and price <= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        long_confirm_count[i] = 0
                        short_confirm_count[i] = 0
                        continue
            
            # Maintain position if signal agrees (1-bar confirmation for exit)
            if prev_side == 1:
                if long_score >= 0.40:
                    # Calculate position size based on signal agreement + regime confidence
                    signal_count = int(long_score * 6)
                    base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    
                    # Regime confidence multiplier
                    regime_conf = 1.0
                    if is_trend_regime:
                        regime_conf = 1.0
                    elif is_range_regime:
                        regime_conf = 0.8
                    else:
                        regime_conf = 0.9
                    
                    target_size = base_target_size * regime_conf
                    target_size = max(min(target_size, 0.35), 0.20)
                    
                    signals[i] = target_size
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    
            elif prev_side == -1:
                if short_score >= 0.40:
                    # Calculate position size based on signal agreement + regime confidence
                    signal_count = int(short_score * 6)
                    base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    
                    # Regime confidence multiplier
                    regime_conf = 1.0
                    if is_trend_regime:
                        regime_conf = 1.0
                    elif is_range_regime:
                        regime_conf = 0.8
                    else:
                        regime_conf = 0.9
                    
                    target_size = base_target_size * regime_conf
                    target_size = max(min(target_size, 0.35), 0.20)
                    
                    signals[i] = -target_size
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
            continue
        
        # Entry logic: require 2-bar confirmation (hysteresis)
        entry_threshold = 0.45
        
        if long_score >= entry_threshold and long_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement + regime confidence
            signal_count = int(long_score * 6)
            base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            
            # Regime confidence multiplier
            regime_conf = 1.0
            if is_trend_regime:
                regime_conf = 1.0
            elif is_range_regime:
                regime_conf = 0.8
            else:
                regime_conf = 0.9
            
            target_size = base_target_size * regime_conf
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = target_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            short_confirm_count[i] = 0
            
        elif short_score >= entry_threshold and short_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement + regime confidence
            signal_count = int(short_score * 6)
            base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            
            # Regime confidence multiplier
            regime_conf = 1.0
            if is_trend_regime:
                regime_conf = 1.0
            elif is_range_regime:
                regime_conf = 0.8
            else:
                regime_conf = 0.9
            
            target_size = base_target_size * regime_conf
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = -target_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            long_confirm_count[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 09:53
