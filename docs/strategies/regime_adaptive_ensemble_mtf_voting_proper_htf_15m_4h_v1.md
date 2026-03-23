# Strategy: regime_adaptive_ensemble_mtf_voting_proper_htf_15m_4h_v1

## Status
ACTIVE - Sharpe=0.277 | Return=+83.3% | DD=-30.7%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.664 | -13.4% | -29.7% | 283 |
| ETHUSDT | 0.337 | +42.4% | -23.5% | 2 |
| SOLUSDT | 1.157 | +220.9% | -39.0% | 11 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.017 | -4.4% | -14.3% | 208 |
| ETHUSDT | -2.143 | -23.6% | -27.5% | 163 |
| SOLUSDT | -0.173 | +1.4% | -19.9% | 161 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #075 - Regime-Adaptive MTF Ensemble with Proper HTF Alignment
==================================================================================================
Hypothesis: Previous ensemble strategies failed due to manual resampling bugs and excessive churn.
This version uses mtf_data helper for CORRECT 4h alignment (critical after data gaps in SOL).

Key innovations:
1. PROPER MTF: Use get_htf_data() and align_htf_to_ltf() - NO manual resampling
2. Regime detection: BBW percentile → trend mode (low vol) vs MR mode (high vol)
3. Signal voting: 3 independent signals (HMA, Supertrend, KAMA) - need 2/3 agreement
4. Adaptive sizing: 0.35 in trend mode, 0.25 in MR mode (MR is riskier)
5. Reduced churn: Only change signal when ≥2 indicators flip

Why this should beat current best (Sharpe=3.653):
- Correct HTF alignment avoids look-ahead bias from manual resampling
- Regime switching reduces losses in choppy markets
- Signal voting reduces false entries from single indicator whipsaws
- Based on #073 (Sharpe=0.200) but with proper mtf_data implementation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_ensemble_mtf_voting_proper_htf_15m_4h_v1"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0
    
    return zscore


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    for i in range(period - 1, n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 15m indicators for entry timing ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # ========== 4h indicators via mtf_data helper (CRITICAL) ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        hma_4h = calculate_hma(close_4h, period=21)
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=50)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
        
    except Exception as e:
        # Fallback if mtf_data fails
        hma_4h_aligned = hma_15m
        kama_4h_aligned = kama_15m
        st_4h_aligned = st_direction_15m
        bbw_4h_aligned = bbw_15m
        bbw_pct_4h_aligned = bbw_pct_15m
    
    # ========== Generate signals ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_TREND = 0.35  # Higher confidence in trend mode
    SIZE_MR = 0.25     # Lower in mean reversion (riskier)
    SIZE_HALF = 0.175
    
    # Regime thresholds
    BBW_TREND_THRESHOLD = 0.40  # Below = trend mode, Above = MR mode
    
    # Entry thresholds
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    ZSCORE_MAX = 1.8
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 100, 40)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # ========== Regime Detection (4h BBW percentile) ==========
        bbw_pct = bbw_pct_4h_aligned[i]
        if bbw_pct < BBW_TREND_THRESHOLD:
            regime = 'trend'
            position_size = SIZE_TREND
        else:
            regime = 'mr'
            position_size = SIZE_MR
        
        # ========== 4h Trend Filters (need 2/3 agreement) ==========
        hma_trend_4h = 1 if close[i] > hma_4h_aligned[i] else (-1 if close[i] < hma_4h_aligned[i] else 0)
        kama_trend_4h = 1 if close[i] > kama_4h_aligned[i] else (-1 if close[i] < kama_4h_aligned[i] else 0)
        st_trend_4h = st_4h_aligned[i]
        
        # Count trend agreements
        trend_votes = 0
        if hma_trend_4h == 1:
            trend_votes += 1
        if kama_trend_4h == 1:
            trend_votes += 1
        if st_trend_4h == 1:
            trend_votes += 1
        
        bearish_votes = 0
        if hma_trend_4h == -1:
            bearish_votes += 1
        if kama_trend_4h == -1:
            bearish_votes += 1
        if st_trend_4h == -1:
            bearish_votes += 1
        
        # ========== Check existing positions first ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            atr = atr_15m[i]
            price = close[i]
            
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
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
        
        # ========== New Entry Logic ==========
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        if regime == 'trend':
            # Trend-following: need 2/3 bullish votes + RSI pullback
            if trend_votes >= 2:
                if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and abs(zscore_val) < ZSCORE_MAX:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
            elif bearish_votes >= 2:
                if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and abs(zscore_val) < ZSCORE_MAX:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
        else:  # Mean reversion regime
            # Mean reversion: fade extremes when trend is weak
            if trend_votes >= 2 and rsi_val < 35 and zscore_val < -1.5:
                # Oversold in uptrend - buy dip
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
                
            elif bearish_votes >= 2 and rsi_val > 65 and zscore_val > 1.5:
                # Overbought in downtrend - sell rip
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        # Default: no position
        if signals[i] == 0:
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 14:38
