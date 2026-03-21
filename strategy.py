#!/usr/bin/env python3
"""
EXPERIMENT #090 - SIMPLIFIED ENSEMBLE VOTING WITH REGIME ADAPTIVE SIZING
==================================================================================================
Hypothesis: Recent ensemble failures (#078-#089) were too complex. This uses 3 simple independent
signal generators that vote, with position size scaled by agreement level. Regime detection via
BBW percentile switches between trend-follow (low vol) and mean-revert (high vol) modes.

Key innovations:
- 3 independent signal generators: (1) Trend (HMA+Supertrend), (2) Momentum (MACD+RSI), (3) Mean Rev (Z-score+BB)
- Vote weighting: 1 signal=0.15, 2 signals=0.25, 3 signals=0.35 (scaled by agreement)
- Regime adaptive: BBW percentile < 30 = trend mode, > 70 = mean-revert mode
- 4h trend filter via mtf_data helper (proper HTF alignment, no look-ahead)
- Volatility scaling: reduce position size when ATR percentile is high
- Conservative sizing: max 0.35 even with full agreement

Why this should work:
- Simpler than failed #078-#089 ensembles
- Uses proven mtf_data helper for proper 4h alignment
- Regime switching avoids trading wrong strategy in wrong market
- Position scaling by agreement reduces noise trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_voting_regime_adaptive_mtf_15m_4h_v3"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback * 100
    
    return percentile


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile for volatility scaling"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        rank = np.sum(window <= atr[i])
        percentile[i] = rank / lookback * 100
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h data using proper mtf_data helper (CRITICAL for no look-ahead)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        hma_4h = calculate_hma(close_4h, period=48)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)[3]
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
        
        use_mtf = True
    except Exception:
        # Fallback if mtf_data not available
        use_mtf = False
        hma_4h_aligned = np.zeros(n)
        st_4h_aligned = np.zeros(n)
        bbw_pct_4h_aligned = np.zeros(n) + 50  # neutral regime
    
    # 15m indicators for entry signals
    atr_15m = calculate_atr(high, low, close, period=14)
    atr_pct_15m = calculate_atr_percentile(atr_15m, lookback=100)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    _, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_middle, bb_lower, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Signal generators
    signals = np.zeros(n)
    
    # Position sizing parameters
    SIZE_BASE = 0.15  # Base size per signal agreement
    SIZE_MAX = 0.35   # Maximum position size
    
    # Regime thresholds
    BBW_TREND_THRESHOLD = 30   # Below = trend regime
    BBW_MEANREV_THRESHOLD = 70 # Above = mean-revert regime
    
    # Volatility scaling
    ATR_HIGH_THRESHOLD = 70    # Reduce size when ATR percentile > 70
    
    # Entry thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    ZSCORE_ENTRY = 1.5
    ZSCORE_EXIT = 0.5
    MACD_HIST_THRESHOLD = 0
    
    first_valid = max(200, 100, 48)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    ATR_STOP_MULT = 2.5
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi = rsi_15m[i]
        zscore = zscore_15m[i]
        st_15m = st_direction_15m[i]
        macd_h = macd_hist[i]
        
        # 4h regime filters
        bbw_regime = bbw_pct_4h_aligned[i] if use_mtf else 50
        hma_4h = hma_4h_aligned[i] if use_mtf else 0
        st_4h = st_4h_aligned[i] if use_mtf else 0
        
        # 4h trend direction
        trend_4h = 0
        if hma_4h > 0:
            if price > hma_4h:
                trend_4h = 1
            elif price < hma_4h:
                trend_4h = -1
        
        # Volatility scaling factor
        vol_scale = 1.0
        if atr_pct_15m[i] > ATR_HIGH_THRESHOLD:
            vol_scale = 0.7  # Reduce size in high vol
        
        # === SIGNAL GENERATOR 1: TREND (HMA + Supertrend) ===
        signal_trend = 0
        if use_mtf and trend_4h != 0:
            # Only trade in direction of 4h trend
            if trend_4h == 1 and st_15m == 1 and price > hma_15m[i]:
                signal_trend = 1
            elif trend_4h == -1 and st_15m == -1 and price < hma_15m[i]:
                signal_trend = -1
        else:
            # No MTF, use 15m only
            if st_15m == 1 and price > hma_15m[i]:
                signal_trend = 1
            elif st_15m == -1 and price < hma_15m[i]:
                signal_trend = -1
        
        # === SIGNAL GENERATOR 2: MOMENTUM (MACD + RSI) ===
        signal_momentum = 0
        if bbw_regime < BBW_MEANREV_THRESHOLD:  # Trend/momentum regime
            if macd_h > MACD_HIST_THRESHOLD and rsi > 50 and rsi < RSI_SHORT_MAX:
                signal_momentum = 1
            elif macd_h < MACD_HIST_THRESHOLD and rsi < 50 and rsi > RSI_LONG_MIN:
                signal_momentum = -1
        else:  # Mean-revert regime
            if rsi < RSI_LONG_MIN:
                signal_momentum = 1
            elif rsi > RSI_SHORT_MAX:
                signal_momentum = -1
        
        # === SIGNAL GENERATOR 3: MEAN REVERSION (Z-score + BB) ===
        signal_meanrev = 0
        if bbw_regime > BBW_MEANREV_THRESHOLD:  # High vol = mean-revert opportunity
            if zscore < -ZSCORE_ENTRY and price < bb_lower[i]:
                signal_meanrev = 1
            elif zscore > ZSCORE_ENTRY and price > bb_upper[i]:
                signal_meanrev = -1
            # Exit signals
            elif abs(zscore) < ZSCORE_EXIT:
                signal_meanrev = 0
        else:  # Low vol = use as confirmation only
            if zscore < -0.5 and price < bb_middle[i]:
                signal_meanrev = 1
            elif zscore > 0.5 and price > bb_middle[i]:
                signal_meanrev = -1
        
        # === VOTE AGGREGATION ===
        vote_sum = signal_trend + signal_momentum + signal_meanrev
        
        # Calculate position size based on agreement
        if vote_sum >= 2:
            target_size = SIZE_BASE * 2 * vol_scale  # 2 signals agree
        elif vote_sum <= -2:
            target_size = -SIZE_BASE * 2 * vol_scale
        elif vote_sum == 3:
            target_size = SIZE_MAX * vol_scale  # All 3 agree
        elif vote_sum == -3:
            target_size = -SIZE_MAX * vol_scale
        elif vote_sum == 1:
            target_size = SIZE_BASE * vol_scale  # 1 signal only
        elif vote_sum == -1:
            target_size = -SIZE_BASE * vol_scale
        else:
            target_size = 0.0
        
        # === POSITION MANAGEMENT (Stoploss/TP) ===
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
                
                # Trail stop at 1R profit
                if current_high > prev_entry + ATR_STOP_MULT * atr:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
                
                # Trail stop at 1R profit
                if current_low < prev_entry - ATR_STOP_MULT * atr:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        continue
            
            # Hold or adjust position based on new signal
            if target_size == 0:
                # Exit if no signal support
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
            else:
                # Keep position if signal agrees
                signals[i] = target_size
                position_side[i] = np.sign(target_size)
                entry_price[i] = prev_entry
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
            continue
        
        # === NEW ENTRY ===
        if target_size != 0:
            # Additional filter: must agree with 4h trend if available
            if use_mtf and trend_4h != 0:
                if np.sign(target_size) != trend_4h:
                    target_size = 0  # Don't trade against 4h trend
            
            if target_size != 0:
                signals[i] = target_size
                position_side[i] = np.sign(target_size)
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals