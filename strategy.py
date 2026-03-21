#!/usr/bin/env python3
"""
EXPERIMENT #053 - Ensemble Voting + Regime Detection + Adaptive Sizing (15m+1h+4h)
==================================================================================================
Hypothesis: Combining 3 independent signal types with regime-based weighting will improve
Sharpe ratio while reducing drawdown. Key innovations:

1. ENSEMBLE VOTING: 3 signal types (Trend, Momentum, Mean-Reversion) vote on direction
   - More agreeing signals = larger position size (adaptive sizing)
   - Reduces false signals from any single indicator

2. REGIME DETECTION: Bollinger Band Width percentile determines market state
   - Low BBW (bottom 30%) = Trend regime → weight trend signals higher
   - High BBW (top 30%) = Mean-reversion regime → weight MR signals higher
   - Middle = Neutral → equal weighting

3. MULTI-TIMEFRAME: 15m entries + 1h trend filter + 4h regime filter
   - Uses mtf_data helper for proper alignment (NO manual resampling!)

4. POSITION SIZING: Discrete levels based on vote count
   - 1 signal: 0.15 | 2 signals: 0.25 | 3 signals: 0.35
   - MAX signal magnitude: 0.35 (critical for drawdown control)

5. RISK MANAGEMENT: 2.5*ATR stoploss, 2R take profit with trailing

Why this should beat #050 (Sharpe=-0.615):
- Ensemble reduces whipsaws from single indicators
- Regime detection avoids trading wrong strategy in wrong market
- Adaptive sizing scales confidence appropriately
- Proper MTF alignment via mtf_data helper
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_voting_regime_adaptive_15m_1h_4h_v2"
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
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
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
    
    # ===== 15m indicators (entry timeframe) =====
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper_15m, bb_mid_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # ===== 1h trend filter (using mtf_data helper) =====
    df_1h = get_htf_data(prices, '1h')
    if df_1h is None or len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    hma_1h = calculate_hma(close_1h, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(high_1h, low_1h, close_1h, period=10, multiplier=3.0)
    macd_1h, _, macd_hist_1h = calculate_macd(close_1h, fast=12, slow=26, signal=9)
    _, _, _, bbw_1h = calculate_bollinger_bands(close_1h, period=20, std_mult=2.0)
    
    # Align 1h indicators to 15m timeframe (auto shift for completed bars)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    st_direction_1h_aligned = align_htf_to_ltf(prices, df_1h, st_direction_1h)
    macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
    bbw_1h_aligned = align_htf_to_ltf(prices, df_1h, bbw_1h)
    
    # ===== 4h regime filter (using mtf_data helper) =====
    df_4h = get_htf_data(prices, '4h')
    if df_4h is None or len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)[3]
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=50)
    
    # Align 4h regime to 15m timeframe
    bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
    
    # ===== Position sizing constants =====
    SIZE_1_SIGNAL = 0.15
    SIZE_2_SIGNALS = 0.25
    SIZE_3_SIGNALS = 0.35
    SIZE_HALF = 0.175
    
    # ===== Thresholds =====
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    ZSCORE_MAX = 2.0
    MACD_MIN = 0.0
    BBW_LOW_REGIME = 0.30  # Bottom 30% = trend regime
    BBW_HIGH_REGIME = 0.70  # Top 30% = mean reversion regime
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 100, 50 * 16)  # Ensure all indicators are ready
    
    # ===== Generate signals =====
    signals = np.zeros(n)
    
    # Position tracking
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN/invalid values
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        
        # ===== REGIME DETECTION (4h BBW percentile) =====
        regime_4h = bbw_pct_4h_aligned[i]
        
        if regime_4h < BBW_LOW_REGIME:
            # Low volatility = TREND regime
            regime_type = 'trend'
        elif regime_4h > BBW_HIGH_REGIME:
            # High volatility = MEAN REVERSION regime
            regime_type = 'mean_reversion'
        else:
            # Middle = NEUTRAL
            regime_type = 'neutral'
        
        # ===== SIGNAL 1: TREND (HMA + Supertrend alignment on 1h) =====
        trend_signal = 0
        hma_1h_val = hma_1h_aligned[i]
        st_1h_val = st_direction_1h_aligned[i]
        
        if not np.isnan(hma_1h_val) and not np.isnan(st_1h_val):
            if price > hma_1h_val and st_1h_val == 1:
                trend_signal = 1
            elif price < hma_1h_val and st_1h_val == -1:
                trend_signal = -1
        
        # Weight trend signal higher in trend regime
        if regime_type == 'trend':
            trend_weight = 1.5
        elif regime_type == 'mean_reversion':
            trend_weight = 0.5
        else:
            trend_weight = 1.0
        
        # ===== SIGNAL 2: MOMENTUM (MACD + RSI on 15m) =====
        momentum_signal = 0
        macd_hist_val = macd_hist_15m[i]
        macd_hist_1h_val = macd_hist_1h_aligned[i]
        rsi_val = rsi_15m[i]
        
        if not np.isnan(macd_hist_val) and not np.isnan(rsi_val):
            if macd_hist_val > MACD_MIN and rsi_val < 60:
                momentum_signal = 1
            elif macd_hist_val < -MACD_MIN and rsi_val > 40:
                momentum_signal = -1
        
        # ===== SIGNAL 3: MEAN REVERSION (Z-score + Bollinger on 15m) =====
        mr_signal = 0
        zscore_val = zscore_15m[i]
        bb_upper_val = bb_upper_15m[i]
        bb_lower_val = bb_lower_15m[i]
        
        if not np.isnan(zscore_val):
            if zscore_val < -1.5 and price < bb_lower_val:
                mr_signal = 1  # Oversold → long
            elif zscore_val > 1.5 and price > bb_upper_val:
                mr_signal = -1  # Overbought → short
        
        # Weight MR signal higher in mean reversion regime
        if regime_type == 'mean_reversion':
            mr_weight = 1.5
        elif regime_type == 'trend':
            mr_weight = 0.5
        else:
            mr_weight = 1.0
        
        # ===== ENSEMBLE VOTING =====
        # Apply regime weights to signals
        weighted_trend = trend_signal * trend_weight
        weighted_momentum = momentum_signal * 1.0
        weighted_mr = mr_signal * mr_weight
        
        # Count agreeing signals (with threshold)
        bullish_votes = 0
        bearish_votes = 0
        
        if weighted_trend > 0.5:
            bullish_votes += 1
        elif weighted_trend < -0.5:
            bearish_votes += 1
        
        if weighted_momentum > 0.5:
            bullish_votes += 1
        elif weighted_momentum < -0.5:
            bearish_votes += 1
        
        if weighted_mr > 0.5:
            bullish_votes += 1
        elif weighted_mr < -0.5:
            bearish_votes += 1
        
        # Determine net signal direction and strength
        net_signal = 0
        vote_count = 0
        
        if bullish_votes >= 2 and bullish_votes > bearish_votes:
            net_signal = 1
            vote_count = bullish_votes
        elif bearish_votes >= 2 and bearish_votes > bullish_votes:
            net_signal = -1
            vote_count = bearish_votes
        
        # Map vote count to position size
        if vote_count == 1:
            signal_size = SIZE_1_SIGNAL
        elif vote_count == 2:
            signal_size = SIZE_2_SIGNALS
        elif vote_count >= 3:
            signal_size = SIZE_3_SIGNALS
        else:
            signal_size = 0.0
        
        # ===== EXISTING POSITION MANAGEMENT =====
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
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
            
            # Hold position if no exit triggered and signal agrees
            if net_signal == prev_side and signal_size > 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Signal changed - close position
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # ===== NEW ENTRY LOGIC =====
        if net_signal != 0 and signal_size > 0:
            signals[i] = net_signal * signal_size
            position_side[i] = net_signal
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals