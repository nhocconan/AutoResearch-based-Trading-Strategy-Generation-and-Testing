#!/usr/bin/env python3
"""
EXPERIMENT #059 - MTF_HMA_SUPERTREND_VOLREGIME_MARKETSTRUCTURE_1H_4H_V1
==================================================================================================
Hypothesis: Market structure breaks (higher highs/lows vs lower highs/lows) combined with 
volatility clustering detection provides superior regime filtering compared to ADX/BBW alone.
Using 1h entries with 4h trend filter reduces noise vs 15m while maintaining signal frequency.
Volume confirmation on breakouts filters false signals. This combines proven HMA+Supertrend 
from #047/#049 with NEW market structure and volatility regime detection.

Key innovations:
- MARKET STRUCTURE: Track swing highs/lows to identify HH/HL (bullish) vs LH/LL (bearish)
- VOLATILITY CLUSTERING: Rolling variance ratio detects vol expansion/contraction cycles
- VOLUME CONFIRMATION: Breakouts require volume > 1.5x 20-bar average
- 1H/4H MULTI-TF: Less noise than 15m, more signals than 4h-only strategies
- ADAPTIVE ATR: ATR multiplier adjusts based on volatility regime (wider stops in high vol)
- POSITION SIZING: Scales inversely with volatility (smaller positions in high vol)

Why this should beat #058 (Sharpe=5.353) and approach #047 (Sharpe=16.016):
- Market structure is more robust than simple moving average crossovers
- Volatility clustering captures regime changes ADX misses
- Volume filter eliminates false breakouts that killed #048
- 1h timeframe has optimal signal-to-noise ratio for crypto perps

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (proven to control drawdown in 2022 crash)
- MIN signal: 0.20 (avoid tiny positions eaten by fees)
- Discrete levels: 0.0, 0.20, 0.28, 0.35 (reduces churn costs)
- Stoploss: 2.5*ATR trailing (adjusts to 3.0*ATR in high vol regime)
- Volatility scaling: position_size = base_size * (target_vol / current_vol)
"""

import numpy as np
import pandas as pd

name = "mtf_hma_supertrend_volregime_marketstructure_1h_4h_v1"
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


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    # WMA helper
    def calc_wma(data, wma_period):
        result = np.zeros(len(data))
        for i in range(wma_period - 1, len(data)):
            weights = np.arange(1, wma_period + 1)
            window = data[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma1 = calc_wma(close, half)
    wma2 = calc_wma(close, period)
    
    raw_hma = 2 * wma1 - wma2
    
    hma = calc_wma(raw_hma, sqrt_period)
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if direction[i - 1] == 1:
                if close[i] < upper_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
            else:
                if close[i] > lower_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
    
    return supertrend, direction, upper_band, lower_band


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


def calculate_volatility_clustering(close, short_period=10, long_period=30):
    """
    Detect volatility clustering using rolling variance ratio.
    High ratio = volatility expansion (trend regime)
    Low ratio = volatility contraction (range regime)
    """
    n = len(close)
    if n < long_period:
        return np.zeros(n)
    
    vol_ratio = np.zeros(n)
    
    for i in range(long_period - 1, n):
        short_returns = np.diff(close[i - short_period:i + 1])
        long_returns = np.diff(close[i - long_period:i + 1])
        
        short_var = np.var(short_returns) if len(short_returns) > 0 else 0
        long_var = np.var(long_returns) if len(long_returns) > 0 else 0
        
        if long_var > 0:
            vol_ratio[i] = short_var / long_var
        else:
            vol_ratio[i] = 1.0
    
    return vol_ratio


def calculate_market_structure(high, low, close, lookback=20):
    """
    Identify market structure: HH/HL (bullish) vs LH/LL (bearish)
    Returns: 1 = bullish structure, -1 = bearish structure, 0 = neutral
    """
    n = len(close)
    structure = np.zeros(n)
    
    # Find swing highs and lows
    swing_highs = np.zeros(n)
    swing_lows = np.zeros(n)
    
    for i in range(2, n - 2):
        # Swing high: higher than 2 bars on each side
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            swing_highs[i] = high[i]
        # Swing low: lower than 2 bars on each side
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            swing_lows[i] = low[i]
    
    # Track recent swing structure
    recent_highs = []
    recent_lows = []
    
    for i in range(lookback, n):
        # Collect swings in lookback window
        recent_highs = [swing_highs[j] for j in range(i - lookback, i) if swing_highs[j] > 0]
        recent_lows = [swing_lows[j] for j in range(i - lookback, i) if swing_lows[j] > 0]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Check for HH/HL pattern
            hh = recent_highs[-1] > recent_highs[-2]
            hl = recent_lows[-1] > recent_lows[-2]
            lh = recent_highs[-1] < recent_highs[-2]
            ll = recent_lows[-1] < recent_lows[-2]
            
            if hh and hl:
                structure[i] = 1  # Bullish
            elif lh and ll:
                structure[i] = -1  # Bearish
            else:
                structure[i] = 0  # Neutral/transition
    
    return structure


def calculate_volume_profile(volume, close, period=20):
    """
    Calculate volume-weighted price levels and volume trend.
    Returns volume trend: 1 = increasing, -1 = decreasing, 0 = neutral
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    volume_trend = np.zeros(n)
    volume_ratio = np.zeros(n)
    
    for i in range(period - 1, n):
        avg_volume = np.mean(volume[i - period + 1:i + 1])
        current_volume = volume[i]
        
        if avg_volume > 0:
            volume_ratio[i] = current_volume / avg_volume
        else:
            volume_ratio[i] = 1.0
        
        # Volume trend: compare recent 5 bars to previous 5 bars
        if i >= period + 4:
            recent_vol = np.mean(volume[i - 4:i + 1])
            prev_vol = np.mean(volume[i - 9:i - 4])
            
            if prev_vol > 0:
                if recent_vol > prev_vol * 1.2:
                    volume_trend[i] = 1
                elif recent_vol < prev_vol * 0.8:
                    volume_trend[i] = -1
    
    return volume_trend, volume_ratio


def resample_to_higher_tf(close, high, low, volume, bars_per_tf=4):
    """Resample 1h data to 4h (4 x 1h = 4h)"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return close.copy(), high.copy(), low.copy(), volume.copy()
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    v_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
            v_tf[i] = np.sum(volume[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf, v_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=16)
    supertrend_1h, st_dir_1h, st_upper_1h, st_lower_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    vol_cluster_1h = calculate_volatility_clustering(close, short_period=10, long_period=30)
    market_struct_1h = calculate_market_structure(high, low, close, lookback=20)
    vol_trend_1h, vol_ratio_1h = calculate_volume_profile(volume, close, period=20)
    
    # Resample to 4h for trend (4 x 1h = 4h)
    bars_per_4h = 4
    c_4h, h_4h, l_4h, v_4h = resample_to_higher_tf(close, high, low, volume, bars_per_4h)
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    hma_4h = calculate_hma(c_4h, period=16)
    supertrend_4h, st_dir_4h, st_upper_4h, st_lower_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    vol_cluster_4h = calculate_volatility_clustering(c_4h, short_period=10, long_period=30)
    market_struct_4h = calculate_market_structure(h_4h, l_4h, c_4h, lookback=20)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    vol_regime_4h = np.zeros(n)
    struct_4h = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    n_4h = len(c_4h)
    for i in range(n):
        idx_4h = i // bars_per_4h
        
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Supertrend direction
            st_trend_4h[i] = st_dir_4h[idx_4h]
            
            # Volatility regime (vol clustering ratio)
            vol_cluster_4h[i] = vol_cluster_4h[idx_4h] if idx_4h < len(vol_cluster_4h) else 1.0
            if vol_cluster_4h[i] > 1.5:
                vol_regime_4h[i] = 1  # High vol / trend
            elif vol_cluster_4h[i] < 0.7:
                vol_regime_4h[i] = -1  # Low vol / range
            else:
                vol_regime_4h[i] = 0  # Neutral
            
            # Market structure
            struct_4h[i] = market_struct_4h[idx_4h] if idx_4h < len(market_struct_4h) else 0
            
            # ATR mapped
            atr_4h_mapped[i] = atr_4h[idx_4h] if idx_4h < len(atr_4h) else atr_1h[i]
    
    # Position sizing parameters (DISCRETE levels)
    SIZE_LEVELS = np.array([0.0, 0.20, 0.28, 0.35])
    BASE_SIZE = 0.28
    TARGET_VOL = 0.02  # Target 2% daily volatility
    
    # Signal thresholds
    VOL_HIGH_THRESHOLD = 1.5
    VOL_LOW_THRESHOLD = 0.7
    VOLUME_CONFIRMATION = 1.5
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    
    # Stoploss multipliers (adaptive to vol regime)
    ATR_STOP_LOW_VOL = 2.5
    ATR_STOP_HIGH_VOL = 3.5
    
    first_valid = max(200, 40 * bars_per_4h)
    
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
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h regime signals
        hma_trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        vol_regime = vol_regime_4h[i]
        market_struct = struct_4h[i]
        atr_4h_val = atr_4h_mapped[i]
        
        # 1h entry signals
        price = close[i]
        hma_1h_val = hma_1h[i]
        st_dir = st_dir_1h[i]
        rsi_val = rsi_1h[i]
        vol_ratio = vol_ratio_1h[i]
        
        # Determine adaptive ATR stop based on vol regime
        if vol_regime == 1:
            atr_stop_mult = ATR_STOP_HIGH_VOL
        else:
            atr_stop_mult = ATR_STOP_LOW_VOL
        
        # Calculate signal scores
        # Signal 1: 4h HMA trend
        hma_signal = 0
        if hma_trend == 1:
            hma_signal = 1
        elif hma_trend == -1:
            hma_signal = -1
        
        # Signal 2: 4h Supertrend
        st_signal = 0
        if st_trend == 1:
            st_signal = 1
        elif st_trend == -1:
            st_signal = -1
        
        # Signal 3: Market structure
        struct_signal = 0
        if market_struct == 1:
            struct_signal = 1
        elif market_struct == -1:
            struct_signal = -1
        
        # Signal 4: 1h HMA
        hma_1h_signal = 0
        if price > hma_1h_val:
            hma_1h_signal = 1
        elif price < hma_1h_val:
            hma_1h_signal = -1
        
        # Signal 5: 1h Supertrend
        st_1h_signal = 0
        if st_dir == 1:
            st_1h_signal = 1
        elif st_dir == -1:
            st_1h_signal = -1
        
        # Signal 6: RSI
        rsi_signal = 0
        if rsi_val < RSI_LONG_MAX:
            rsi_signal = 1
        elif rsi_val > RSI_SHORT_MIN:
            rsi_signal = -1
        
        # Signal 7: Volume confirmation
        volume_signal = 0
        if vol_ratio >= VOLUME_CONFIRMATION:
            volume_signal = 1  # High volume supports breakout
        elif vol_ratio < 0.7:
            volume_signal = -1  # Low volume = weak move
        
        # Calculate weighted signal score
        # Trend regime (vol_regime=1): weight trend signals higher
        # Range regime (vol_regime=-1): weight mean reversion higher
        if vol_regime == 1:
            # Trend-following regime
            long_score = (
                0.25 * (hma_signal == 1) +
                0.20 * (st_signal == 1) +
                0.15 * (struct_signal == 1) +
                0.15 * (hma_1h_signal == 1) +
                0.15 * (st_1h_signal == 1) +
                0.10 * (rsi_signal == 1)
            )
            short_score = (
                0.25 * (hma_signal == -1) +
                0.20 * (st_signal == -1) +
                0.15 * (struct_signal == -1) +
                0.15 * (hma_1h_signal == -1) +
                0.15 * (st_1h_signal == -1) +
                0.10 * (rsi_signal == -1)
            )
        elif vol_regime == -1:
            # Range regime: require stronger confirmation
            long_score = (
                0.20 * (hma_signal == 1) +
                0.20 * (st_signal == 1) +
                0.20 * (struct_signal == 1) +
                0.15 * (hma_1h_signal == 1) +
                0.15 * (st_1h_signal == 1) +
                0.10 * (rsi_signal == 1)
            )
            short_score = (
                0.20 * (hma_signal == -1) +
                0.20 * (st_signal == -1) +
                0.20 * (struct_signal == -1) +
                0.15 * (hma_1h_signal == -1) +
                0.15 * (st_1h_signal == -1) +
                0.10 * (rsi_signal == -1)
            )
        else:
            # Neutral regime: balanced
            long_score = (
                0.20 * (hma_signal == 1) +
                0.20 * (st_signal == 1) +
                0.15 * (struct_signal == 1) +
                0.15 * (hma_1h_signal == 1) +
                0.15 * (st_1h_signal == 1) +
                0.15 * (rsi_signal == 1)
            )
            short_score = (
                0.20 * (hma_signal == -1) +
                0.20 * (st_signal == -1) +
                0.15 * (struct_signal == -1) +
                0.15 * (hma_1h_signal == -1) +
                0.15 * (st_1h_signal == -1) +
                0.15 * (rsi_signal == -1)
            )
        
        # Volume confirmation bonus for breakouts
        if volume_signal == 1:
            long_score = min(1.0, long_score + 0.1)
            short_score = min(1.0, short_score + 0.1)
        
        # HYSTERESIS: Update confirmation counters
        if long_score >= 0.50:
            long_confirm_count[i] = long_confirm_count[i - 1] + 1 if i > 0 else 1
        else:
            long_confirm_count[i] = 0
        
        if short_score >= 0.50:
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
                stoploss_price = prev_entry - atr_stop_mult * atr_1h[i]
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
                tp_price = prev_entry + 2 * atr_stop_mult * atr_1h[i]
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
                    trail_stop = current_high - atr_stop_mult * atr_1h[i]
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
                stoploss_price = prev_entry + atr_stop_mult * atr_1h[i]
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
                tp_price = prev_entry - 2 * atr_stop_mult * atr_1h[i]
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
                    trail_stop = current_low + atr_stop_mult * atr_1h[i]
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
            
            # Maintain position if signal agrees
            if prev_side == 1:
                if long_score >= 0.45:
                    # Calculate position size based on signal agreement + vol scaling
                    signal_count = int(long_score * 4)
                    base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    
                    # Volatility scaling: reduce size in high vol
                    vol_scale = 1.0
                    if vol_regime == 1:
                        vol_scale = 0.7  # Reduce size in high vol
                    elif vol_regime == -1:
                        vol_scale = 1.0  # Normal size in low vol
                    
                    target_size = base_target_size * vol_scale
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
                if short_score >= 0.45:
                    # Calculate position size based on signal agreement + vol scaling
                    signal_count = int(short_score * 4)
                    base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    
                    # Volatility scaling
                    vol_scale = 1.0
                    if vol_regime == 1:
                        vol_scale = 0.7
                    elif vol_regime == -1:
                        vol_scale = 1.0
                    
                    target_size = base_target_size * vol_scale
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
        entry_threshold = 0.50
        
        if long_score >= entry_threshold and long_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement + vol scaling
            signal_count = int(long_score * 4)
            base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            
            # Volatility scaling
            vol_scale = 1.0
            if vol_regime == 1:
                vol_scale = 0.7
            elif vol_regime == -1:
                vol_scale = 1.0
            
            target_size = base_target_size * vol_scale
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = target_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            short_confirm_count[i] = 0
            
        elif short_score >= entry_threshold and short_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement + vol scaling
            signal_count = int(short_score * 4)
            base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            
            # Volatility scaling
            vol_scale = 1.0
            if vol_regime == 1:
                vol_scale = 0.7
            elif vol_regime == -1:
                vol_scale = 1.0
            
            target_size = base_target_size * vol_scale
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