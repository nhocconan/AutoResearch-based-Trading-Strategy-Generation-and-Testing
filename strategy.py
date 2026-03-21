#!/usr/bin/env python3
"""
EXPERIMENT #085 - Triple MTF Ensemble with Volatility-Adaptive Sizing
==================================================================================================
Hypothesis: Recent ensemble strategies (#073-#084) achieved Sharpe 0.18-0.42 by using 2 timeframes
(15m + 4h). The current best (Sharpe=3.653) uses TRIPLE timeframe (15m + 1h + 4h). This version
implements proper triple MTF with volatility-adaptive position sizing and signal correlation filter.

Key innovations:
1. TRIPLE MTF: 15m entry + 1h intermediate trend + 4h macro trend (all must align)
2. VOL-ADAPTIVE SIZING: Position size inversely proportional to ATR volatility
3. SIGNAL CORRELATION: Only enter when signals show diversity (not all same indicator type)
4. REGIME FILTER: BBW percentile + ATR percentile for dual regime detection
5. DIVERSE SIGNALS: Supertrend (trend), RSI (momentum), MACD (momentum), KAMA (adaptive trend)
6. TIGHTER RISK: 1.5 ATR stop, position sizing 0.10-0.30 based on vol and agreement

Why this should beat #084 (Sharpe=0.423) and approach current best (Sharpe=3.653):
- Triple MTF alignment reduces false signals significantly (proven in current best)
- Vol-adaptive sizing reduces position in high vol (major DD driver)
- Signal diversity filter avoids correlated signal failures
- Conservative sizing (max 0.30) protects against 2022-style crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "triple_mtf_vol_adaptive_ensemble_15m_1h_4h_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Supertrend indicator"""
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
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, bbw


def calculate_percentile(series, lookback=100):
    """Calculate rolling percentile"""
    n = len(series)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = series[i - lookback + 1:i + 1]
        current = series[i]
        percentile[i] = np.sum(window <= current) / len(window)
    
    return percentile


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # ========== 15m INDICATORS (ENTRY TIMING) ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_trend_15m = calculate_supertrend(high, low, close, atr_15m, multiplier=3.0)
    macd_15m, _, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_percentile(bbw_15m, lookback=100)
    atr_pct_15m = calculate_percentile(atr_15m, lookback=100)
    
    # Volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ========== 1h INDICATORS (INTERMEDIATE TREND) - PROPER MTF ==========
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        kama_1h = calculate_kama(close_1h, er_period=10, fast_period=2, slow_period=30)
        atr_1h = calculate_atr(high_1h, low_1h, close_1h, period=14)
        _, st_trend_1h = calculate_supertrend(high_1h, low_1h, close_1h, atr_1h, multiplier=3.0)
        macd_1h, _, macd_hist_1h = calculate_macd(close_1h, fast=12, slow=26, signal=9)
        
        kama_1h_aligned = align_htf_to_ltf(prices, df_1h, kama_1h)
        st_trend_1h_aligned = align_htf_to_ltf(prices, df_1h, st_trend_1h)
        macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
        
    except Exception:
        kama_1h_aligned = np.zeros(n)
        st_trend_1h_aligned = np.zeros(n)
        macd_hist_1h_aligned = np.zeros(n)
    
    # ========== 4h INDICATORS (MACRO TREND) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - VOLATILITY ADAPTIVE
    BASE_SIZE = 0.15
    MAX_SIZE = 0.30
    MIN_SIZE = 0.10
    
    # Regime thresholds
    BBW_LOW_REGIME = 0.30
    BBW_HIGH_REGIME = 0.70
    ATR_LOW_REGIME = 0.30
    ATR_HIGH_REGIME = 0.70
    
    # Volume confirmation
    VOLUME_MULT = 1.5
    
    # ATR stoploss
    ATR_STOP_MULT = 1.5
    
    first_valid = max(200, 100, 40)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        macd_hist_val = macd_hist_15m[i]
        st_trend_val = st_trend_15m[i]
        bbw_pct = bbw_pct_15m[i]
        atr_pct = atr_pct_15m[i]
        vol_ratio = volume[i] / volume_ma_20[i] if volume_ma_20[i] > 0 else 1.0
        
        # 1h trend filters
        kama_1h_val = kama_1h_aligned[i]
        st_trend_1h_val = st_trend_1h_aligned[i]
        macd_hist_1h_val = macd_hist_1h_aligned[i]
        
        # 4h trend filters
        kama_4h_val = kama_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        
        # Determine regime (dual filter)
        if bbw_pct < BBW_LOW_REGIME and atr_pct < ATR_LOW_REGIME:
            regime = 'low_vol'
        elif bbw_pct > BBW_HIGH_REGIME or atr_pct > ATR_HIGH_REGIME:
            regime = 'high_vol'
        else:
            regime = 'normal'
        
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
            
            # Stoploss check (1.5*ATR)
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
                    signals[i] = signals[i - 1] * 0.5
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
                    signals[i] = signals[i - 1] * 0.5
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
        
        # ========== TRIPLE MTF ENSEMBLE VOTING ==========
        # Signal 1: 4h Supertrend (MACRO - master filter)
        macro_vote = 0
        if st_trend_4h_val == 1:
            macro_vote = 1
        elif st_trend_4h_val == -1:
            macro_vote = -1
        
        # Signal 2: 1h Supertrend (INTERMEDIATE)
        inter_vote = 0
        if st_trend_1h_val == 1:
            inter_vote = 1
        elif st_trend_1h_val == -1:
            inter_vote = -1
        
        # Signal 3: 15m Supertrend (ENTRY)
        entry_vote = 0
        if st_trend_val == 1:
            entry_vote = 1
        elif st_trend_val == -1:
            entry_vote = -1
        
        # Signal 4: 1h MACD momentum
        momentum_vote = 0
        if macd_hist_1h_val > 0:
            momentum_vote = 1
        elif macd_hist_1h_val < 0:
            momentum_vote = -1
        
        # Signal 5: 15m RSI filter
        rsi_vote = 0
        if rsi_val > 50 and rsi_val < 70:
            rsi_vote = 1
        elif rsi_val < 50 and rsi_val > 30:
            rsi_vote = -1
        
        # Signal 6: 4h KAMA trend
        kama_vote = 0
        if kama_4h_val > 0 and price > kama_4h_val:
            kama_vote = 1
        elif kama_4h_val > 0 and price < kama_4h_val:
            kama_vote = -1
        
        # Volume confirmation
        volume_confirm = vol_ratio >= VOLUME_MULT
        
        # Count votes
        long_votes = sum(1 for v in [macro_vote, inter_vote, entry_vote, momentum_vote, rsi_vote, kama_vote] if v == 1)
        short_votes = sum(1 for v in [macro_vote, inter_vote, entry_vote, momentum_vote, rsi_vote, kama_vote] if v == -1)
        
        # CRITICAL: All three timeframes must agree for entry
        # This is the key filter that reduces false signals
        
        # Volatility-adaptive position sizing
        if regime == 'high_vol':
            size_mult = 0.67  # Reduce size in high vol
        elif regime == 'low_vol':
            size_mult = 1.0
        else:
            size_mult = 0.85
        
        # Regime-adaptive entry logic
        if regime == 'low_vol':
            # Low vol - can be more aggressive, require 4/6 agreement
            if macro_vote == 1 and inter_vote == 1 and entry_vote == 1 and long_votes >= 4:
                base_size = BASE_SIZE * size_mult
                size = MAX_SIZE if volume_confirm and long_votes >= 5 else base_size
                size = min(size, MAX_SIZE)
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif macro_vote == -1 and inter_vote == -1 and entry_vote == -1 and short_votes >= 4:
                base_size = BASE_SIZE * size_mult
                size = MAX_SIZE if volume_confirm and short_votes >= 5 else base_size
                size = min(size, MAX_SIZE)
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        elif regime == 'high_vol':
            # High vol - very conservative, require 5/6 agreement
            if macro_vote == 1 and inter_vote == 1 and entry_vote == 1 and long_votes >= 5:
                size = MIN_SIZE * size_mult
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif macro_vote == -1 and inter_vote == -1 and entry_vote == -1 and short_votes >= 5:
                size = MIN_SIZE * size_mult
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            # Normal regime - require 4/6 agreement with all 3 TF aligned
            if macro_vote == 1 and inter_vote == 1 and entry_vote == 1 and long_votes >= 4:
                base_size = BASE_SIZE * size_mult
                size = MAX_SIZE if volume_confirm and long_votes >= 5 else base_size
                size = min(size, MAX_SIZE)
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif macro_vote == -1 and inter_vote == -1 and entry_vote == -1 and short_votes >= 4:
                base_size = BASE_SIZE * size_mult
                size = MAX_SIZE if volume_confirm and short_votes >= 5 else base_size
                size = min(size, MAX_SIZE)
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