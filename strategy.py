#!/usr/bin/env python3
"""
EXPERIMENT #097 - MTF HMA+RSI+BBW Regime Filter with Proper HTF Alignment (15m+4h v2)
==================================================================================================
Hypothesis: Experiments #086-#096 failed due to overly complex ensemble voting and improper MTF alignment.
The current best (#040) uses 15m+1h with Sharpe=5.4, but we can beat it with 15m+4h + regime filter.

Key changes from #040:
- Use mtf_data helper (get_htf_data, align_htf_to_ltf) for PROPER 4h alignment (no synthetic resampling)
- Regime filter: BBW percentile to detect low-vol (trend) vs high-vol (mean-revert) regimes
- Simpler signal combination: HMA trend (4h) + RSI pullback (15m) + BBW regime (4h)
- Discrete position sizing: 0.0, ±0.20, ±0.35 only (reduce churn costs)
- ATR stoploss: 2.5*ATR with trailing at 1.5R profit
- ADX filter removed (was causing too many missed trades in #094)

Why this should beat #040:
- 4h trend is more stable than 1h (fewer whipsaws)
- BBW regime filter adapts to market conditions (trend vs mean-revert)
- Proper HTF alignment eliminates look-ahead bias from synthetic resampling
- Based on lessons from #087 (Sharpe=0.028) and #093 (Sharpe=0.189) with proper HTF
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_bbw_regime_proper_htf_15m_4h_v2"
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
    
    wma1 = pd.Series(close).rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1) / np.sum(np.arange(1, len(x) + 1))),
        min_periods=half_period
    ).values
    
    wma2 = pd.Series(close).rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1) / np.sum(np.arange(1, len(x) + 1))),
        min_periods=period
    ).values
    
    hma_raw = 2 * wma1 - wma2
    
    hma = pd.Series(hma_raw).rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1) / np.sum(np.arange(1, len(x) + 1))),
        min_periods=sqrt_period
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
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
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h indicators for trend and regime (using PROPER mtf_data helper)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        hma_4h_raw = calculate_hma(close_4h, period=21)
        _, _, _, bbw_4h_raw = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_pct_4h_raw = calculate_bbw_percentile(bbw_4h_raw, lookback=100)
        
        # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
        hma_4h = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
        bbw_4h = align_htf_to_ltf(prices, df_4h, bbw_4h_raw)
        bbw_pct_4h = align_htf_to_ltf(prices, df_4h, bbw_pct_4h_raw)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h = np.zeros(n)
        bbw_4h = np.zeros(n)
        bbw_pct_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.10
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.5
    
    # BBW percentile regime thresholds
    BBW_LOW_VOL = 0.30  # Below 30th percentile = low vol (trend regime)
    BBW_HIGH_VOL = 0.70  # Above 70th percentile = high vol (mean-revert regime)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    ATR_TRAIL_MULT = 1.5
    
    first_valid = max(200, 14 * 2, 20, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 0
        if hma_4h[i] > 0 and close[i] > hma_4h[i]:
            trend_4h = 1
        elif hma_4h[i] > 0 and close[i] < hma_4h[i]:
            trend_4h = -1
        
        # 4h regime filter (BBW percentile)
        regime = 0  # 0=neutral, 1=trend (low vol), 2=mean-revert (high vol)
        if bbw_pct_4h[i] < BBW_LOW_VOL:
            regime = 1  # Low volatility - trend following
        elif bbw_pct_4h[i] > BBW_HIGH_VOL:
            regime = 2  # High volatility - mean reversion
        
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        
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
            
            # Stoploss check (2.5*ATR)
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
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry + 2.5 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1.5R profit
                if prev_tp:
                    trail_stop = current_high - ATR_TRAIL_MULT * atr
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
                
                # Take profit check (2.5R) - reduce to half
                tp_price = prev_entry - 2.5 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1.5R profit
                if prev_tp:
                    trail_stop = current_low + ATR_TRAIL_MULT * atr
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
        
        # Entry logic based on regime
        if regime == 1:  # Low volatility - trend following
            # Long: 4h HMA bullish + 15m RSI pullback + Z-score not extreme
            if trend_4h == 1 and (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX) and abs(zscore_val) < ZSCORE_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            
            # Short: 4h HMA bearish + 15m RSI pullback + Z-score not extreme
            elif trend_4h == -1 and (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX) and abs(zscore_val) < ZSCORE_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        elif regime == 2:  # High volatility - mean reversion
            # Long: RSI oversold + Z-score negative extreme
            if rsi_val < 30 and zscore_val < -1.5:
                signals[i] = SIZE_QUARTER
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            
            # Short: RSI overbought + Z-score positive extreme
            elif rsi_val > 70 and zscore_val > 1.5:
                signals[i] = -SIZE_QUARTER
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:  # Neutral regime - stay flat or reduce position
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals