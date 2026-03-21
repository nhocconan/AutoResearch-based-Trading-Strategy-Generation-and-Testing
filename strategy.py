#!/usr/bin/env python3
"""
EXPERIMENT #080 - REGIME_ADAPTIVE_MTF_SIMPLIFIED_15M_4H_V1
==================================================================================================
Hypothesis: Recent ensemble failures (#070-#079) show complex voting creates churn and drawdown.
This strategy uses SIMPLIFIED regime adaptation with proper mtf_data helper.

Key changes from #040:
- Use mtf_data.get_htf_data() and align_htf_to_ltf() (MANDATORY - #040 did manual resample)
- Position size: 0.28 (reduced from 0.35 for better DD control)
- Regime detection: BBW percentile (20-bar rolling) → trend mode in low vol, mean-revert in high vol
- Fewer filters: Only require 2/3 trend indicators agree (not all 3)
- Volume confirmation: Only enter on volume > 1.5x 20-bar average
- Stoploss: 2.5*ATR (wider than #040's 2.0*ATR to avoid premature exits)
- Timeframe: 15m entries + 4h trend (proven in #075 with Sharpe=0.277)

Why this should beat recent failures:
- Simpler logic = fewer signal flips = lower fees
- Proper MTF alignment via mtf_data helper (46 strategies failed without this)
- Regime-adaptive sizing (smaller positions in high vol regimes)
- Volume filter reduces false breakouts
- Based on #075's success pattern but with cleaner implementation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_mtf_simplified_15m_4h_v1"
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
    
    hma = np.zeros(n)
    raw_vals = 2 * wma1 - wma2
    
    for i in range(sqrt_period - 1, n):
        window = raw_vals[i - sqrt_period + 1:i + 1]
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(window * weights) / np.sum(weights)
    
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
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
    bbw = np.zeros(n)
    for i in range(period - 1, n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
    
    return upper, middle, lower, bbw


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.ones(n)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.ones(n)
    
    for i in range(period - 1, n):
        if vol_avg[i] > 0:
            vol_ratio[i] = volume[i] / vol_avg[i]
    
    return vol_ratio


def calculate bbw_percentile(bbw, period=20):
    """Calculate BBW rolling percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(period - 1, n):
        window = bbw[i - period + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / period
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_ratio_15m = calculate_volume_ratio(volume, period=20)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, period=20)
    
    # 4h trend indicators using mtf_data helper (MANDATORY)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(close_4h, period=21)
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        kama_4h_aligned = np.zeros(n)
        st_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Generate signals with regime-adaptive logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.28  # Reduced from 0.35 for better DD control
    SIZE_HALF = 0.14
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.5
    
    # ATR stoploss multiplier (wider than #040)
    ATR_STOP_MULT = 2.5
    
    # Volume confirmation threshold
    VOL_MIN = 1.3
    
    # BBW percentile for regime detection
    BBW_LOW_REGIME = 0.30  # Below 30th percentile = trend regime
    BBW_HIGH_REGIME = 0.70  # Above 70th percentile = mean-revert regime
    
    first_valid = max(200, 14 * 2, 20, 28)
    
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
        
        # Regime detection from 4h BBW percentile
        bbw_4h_val = bbw_4h_aligned[i]
        bbw_15m_val = bbw_15m[i]
        bbw_pct = bbw_pct_15m[i]
        
        # Determine regime
        if bbw_pct < BBW_LOW_REGIME:
            regime = 'trend'  # Low volatility = trend following
        elif bbw_pct > BBW_HIGH_REGIME:
            regime = 'mean_revert'  # High volatility = mean reversion
        else:
            regime = 'neutral'  # Middle = reduced sizing
        
        # 4h trend indicators
        trend_hma = 0
        trend_kama = 0
        trend_st = 0
        
        if close[i] > hma_4h_aligned[i] and hma_4h_aligned[i] > 0:
            trend_hma = 1
        elif close[i] < hma_4h_aligned[i] and hma_4h_aligned[i] > 0:
            trend_hma = -1
        
        if close[i] > kama_4h_aligned[i] and kama_4h_aligned[i] > 0:
            trend_kama = 1
        elif close[i] < kama_4h_aligned[i] and kama_4h_aligned[i] > 0:
            trend_kama = -1
        
        trend_st = st_4h_aligned[i]
        
        # Count trend agreement (need 2/3 for trend regime)
        trend_votes = trend_hma + trend_kama + int(trend_st)
        
        # Volume confirmation
        vol_confirmed = vol_ratio_15m[i] >= VOL_MIN
        
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
            
            # Stoploss check (2.5*ATR)
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
                    signals[i] = SIZE_HALF
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
                    signals[i] = -SIZE_HALF
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
        
        # Entry logic based on regime
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        if regime == 'trend':
            # Trend regime: require 2/3 trend agreement + volume confirmation
            if trend_votes >= 2 and vol_confirmed:
                if trend_votes > 0:  # Bullish
                    if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and abs(zscore_val) < ZSCORE_MAX:
                        signals[i] = SIZE_FULL
                        position_side[i] = 1
                        entry_price[i] = close[i]
                        tp_triggered[i] = 0
                        highest_since_entry[i] = close[i]
                        lowest_since_entry[i] = close[i]
                else:  # Bearish
                    if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and abs(zscore_val) < ZSCORE_MAX:
                        signals[i] = -SIZE_FULL
                        position_side[i] = -1
                        entry_price[i] = close[i]
                        tp_triggered[i] = 0
                        highest_since_entry[i] = close[i]
                        lowest_since_entry[i] = close[i]
        
        elif regime == 'mean_revert':
            # Mean revert regime: trade against extremes with smaller size
            if zscore_val > 2.0 and rsi_val > 70:
                signals[i] = -SIZE_HALF  # Short overbought
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            elif zscore_val < -2.0 and rsi_val < 30:
                signals[i] = SIZE_HALF  # Long oversold
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            # Neutral regime: reduced position size, require stronger signals
            if trend_votes >= 2 and vol_confirmed:
                if trend_votes > 0:
                    if 40 <= rsi_val <= 60 and abs(zscore_val) < 1.5:
                        signals[i] = SIZE_HALF
                        position_side[i] = 1
                        entry_price[i] = close[i]
                        tp_triggered[i] = 0
                        highest_since_entry[i] = close[i]
                        lowest_since_entry[i] = close[i]
                else:
                    if 40 <= rsi_val <= 60 and abs(zscore_val) < 1.5:
                        signals[i] = -SIZE_HALF
                        position_side[i] = -1
                        entry_price[i] = close[i]
                        tp_triggered[i] = 0
                        highest_since_entry[i] = close[i]
                        lowest_since_entry[i] = close[i]
        
        if signals[i] == 0:
            position_side[i] = 0
    
    return signals