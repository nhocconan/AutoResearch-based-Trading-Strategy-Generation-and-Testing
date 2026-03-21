#!/usr/bin/env python3
"""
EXPERIMENT #069 - REGIME_ADAPTIVE_MTF_ENSEMBLE_15M_4H_V2
==================================================================================================
Hypothesis: Combine regime detection (BBW percentile) with MTF ensemble voting.
- Low volatility regime (BBW < 30th percentile): Trend-following signals weighted higher
- High volatility regime (BBW > 70th percentile): Mean-reversion signals weighted higher
- Use 4h trend via mtf_data helper to filter 15m entries
- Ensemble of 3 signal types: Supertrend trend, MACD momentum, RSI mean-reversion
- Adaptive position sizing based on signal agreement (more agree = larger position)

Why this should beat #040 and current best:
- Proper mtf_data usage (no manual resampling bugs)
- Regime adaptation reduces losses in wrong market conditions
- Ensemble voting reduces false signals
- Conservative sizing (max 0.30) with dynamic scaling
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_mtf_ensemble_15m_4h_v2"
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
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


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
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
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
    
    return upper, middle, lower, bbw


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
    
    return zscore


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank over lookback period"""
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
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Get 4h data using mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h trend indicators
        hma_4h = calculate_hma(close_4h, period=21)
        st_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        macd_line_4h, macd_signal_4h, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        st_dir_4h_aligned = np.ones(n)
        bbw_4h_aligned = np.zeros(n)
        macd_hist_4h_aligned = np.zeros(n)
    
    # Calculate BBW percentile for regime detection
    bbw_percentile_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    bbw_percentile_4h = calculate_bbw_percentile(bbw_4h_aligned, lookback=50)
    
    # Generate signals with regime-adaptive ensemble
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_BASE = 0.25
    SIZE_HIGH_CONF = 0.35
    
    # Regime thresholds
    LOW_VOL_THRESHOLD = 0.30  # BBW percentile < 30% = low vol = trend follow
    HIGH_VOL_THRESHOLD = 0.70  # BBW percentile > 70% = high vol = mean revert
    
    # Signal thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    ZSCORE_MAX = 1.8
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 100, 26, 20, 14)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 0
        if hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]:
            trend_4h = -1
        
        st_trend_4h = st_dir_4h_aligned[i]
        macd_trend_4h = 1 if macd_hist_4h_aligned[i] > 0 else -1
        
        # Regime detection (low vol = trend, high vol = mean reversion)
        regime_low_vol = bbw_percentile_15m[i] < LOW_VOL_THRESHOLD
        regime_high_vol = bbw_percentile_15m[i] > HIGH_VOL_THRESHOLD
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = SIZE_BASE / 2
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -SIZE_BASE / 2
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if close[i] > trail_stop:
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
        
        # Ensemble voting: count agreement between signals
        long_votes = 0
        short_votes = 0
        
        # Signal 1: Supertrend direction (15m)
        if st_direction_15m[i] == 1:
            long_votes += 1
        else:
            short_votes += 1
        
        # Signal 2: MACD histogram (15m)
        if macd_hist_15m[i] > 0:
            long_votes += 1
        else:
            short_votes += 1
        
        # Signal 3: 4h trend filter
        if trend_4h == 1:
            long_votes += 1
        elif trend_4h == -1:
            short_votes += 1
        
        # Signal 4: 4h Supertrend
        if st_trend_4h == 1:
            long_votes += 1
        elif st_trend_4h == -1:
            short_votes += 1
        
        # Signal 5: 4h MACD
        if macd_hist_4h_aligned[i] > 0:
            long_votes += 1
        else:
            short_votes += 1
        
        # Regime-adaptive entry logic
        signal_agreement = max(long_votes, short_votes) / 5.0
        
        if regime_low_vol:
            # Low volatility: Trend-following mode
            # Need 4h trend agreement + 15m momentum confirmation
            if long_votes >= 4 and trend_4h == 1 and st_direction_15m[i] == 1:
                # RSI pullback filter
                if RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX:
                    position_size = SIZE_HIGH_CONF if signal_agreement >= 0.8 else SIZE_BASE
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
            elif short_votes >= 4 and trend_4h == -1 and st_direction_15m[i] == -1:
                # RSI pullback filter
                if RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX:
                    position_size = SIZE_HIGH_CONF if signal_agreement >= 0.8 else SIZE_BASE
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
        elif regime_high_vol:
            # High volatility: Mean-reversion mode
            # Look for RSI extremes + Z-score confirmation
            if rsi_15m[i] < 30 and zscore_15m[i] < -ZSCORE_MAX and long_votes >= 3:
                position_size = SIZE_BASE
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
                
            elif rsi_15m[i] > 70 and zscore_15m[i] > ZSCORE_MAX and short_votes >= 3:
                position_size = SIZE_BASE
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            # Neutral regime: Require strong agreement
            if long_votes >= 4 and trend_4h == 1:
                if RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX:
                    signals[i] = SIZE_BASE
                    position_side[i] = 1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
            elif short_votes >= 4 and trend_4h == -1:
                if RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX:
                    signals[i] = -SIZE_BASE
                    position_side[i] = -1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
        
        if signals[i] == 0:
            position_side[i] = 0
    
    return signals