#!/usr/bin/env python3
"""
EXPERIMENT #058 - Ensemble Voting + Regime Detection + Adaptive Sizing (1h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.563) uses BB regime + Supertrend + RSI on 1h.
This strategy improves by:
1. Ensemble voting: 3 independent signals (Trend, Momentum, Mean-Reversion) vote on direction
2. Regime-adaptive: BB Width percentile determines which signals to trust (trend vs mean-revert)
3. Adaptive sizing: More agreement = larger position (0.20 single, 0.30 double, 0.35 triple)
4. 4h HMA trend filter for directional bias (proven in #047, #053)
5. Z-score filter to avoid extreme overbought/oversold entries

Why this should beat current best (Sharpe=0.563):
- Ensemble reduces false signals (need 2/3 agreement vs single indicator)
- Regime detection avoids trend-following in chop and mean-reversion in strong trends
- Adaptive sizing maximizes returns on high-conviction setups
- 1h primary captures intraday moves with fewer whipsaws than 30m
- Conservative base sizing (0.20) controls drawdown during uncertain regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_regime_adaptive_voting_1h_4h_v2"
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


def calculate_hma(close, period=21):
    """
    Hull Moving Average - faster response than EMA with less lag
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close_series, half_period)
    wma_full = wma(close_series, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma.values


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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma  # Normalized band width
    
    return upper, lower, sma, band_width


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - sma[mask]) / std[mask]
    
    return zscore


def calculate_bb_width_percentile(band_width, lookback=100):
    """Calculate rolling percentile of BB Width for regime detection"""
    n = len(band_width)
    if n < lookback:
        return np.zeros(n)
    
    percentile = np.zeros(n)
    for i in range(lookback - 1, n):
        window = band_width[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= band_width[i]) / len(valid) * 100
        else:
            percentile[i] = 50
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper_1h, bb_lower_1h, bb_sma_1h, bb_width_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    zscore_1h = calculate_zscore(close, period=20)
    bb_percentile_1h = calculate_bb_width_percentile(bb_width_1h, lookback=100)
    
    # HMA for trend direction
    hma_1h = calculate_hma(close, period=21)
    hma_1h_fast = calculate_hma(close, period=9)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h_raw = calculate_hma(close_4h, period=21)
        
        # 4h RSI for overbought/oversold context
        rsi_4h_raw = calculate_rsi(close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - ADAPTIVE based on signal agreement
    SIZE_SINGLE = 0.20   # 1 signal agrees (low conviction)
    SIZE_DOUBLE = 0.30   # 2 signals agree (medium conviction)
    SIZE_TRIPLE = 0.35   # 3 signals agree (high conviction)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # Regime thresholds
    BB_PERCENTILE_LOW = 30    # Low volatility = trend regime
    BB_PERCENTILE_HIGH = 70   # High volatility = mean reversion regime
    
    # Signal thresholds
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    ZSCORE_EXTREME = 2.0
    
    first_valid = max(200, 150)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        macd_hist = macd_hist_1h[i]
        zscore_val = zscore_1h[i]
        bb_pct = bb_percentile_1h[i]
        
        # Price position in BB
        bb_position = (price - bb_lower_1h[i]) / (bb_upper_1h[i] - bb_lower_1h[i]) if (bb_upper_1h[i] - bb_lower_1h[i]) > 0 else 0.5
        
        # HMA trend
        hma_slope = hma_1h[i] - hma_1h[i - 1] if i > 0 else 0
        hma_fast_slope = hma_1h_fast[i] - hma_1h_fast[i - 1] if i > 0 else 0
        
        # 4h trend filters
        hma_4h_val = hma_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0:
            if price > hma_4h_val:
                trend_4h = 1
            elif price < hma_4h_val:
                trend_4h = -1
        
        # Determine regime
        is_trend_regime = bb_pct < BB_PERCENTILE_LOW
        is_mr_regime = bb_pct > BB_PERCENTILE_HIGH
        is_neutral_regime = not is_trend_regime and not is_mr_regime
        
        # ========== SIGNAL VOTING SYSTEM ==========
        # Each signal votes: +1 (long), -1 (short), 0 (neutral)
        trend_vote = 0
        momentum_vote = 0
        mr_vote = 0
        
        # SIGNAL 1: TREND (HMA slope + price position + 4h alignment)
        if is_trend_regime or is_neutral_regime:
            if hma_slope > 0 and hma_fast_slope > 0 and price > hma_1h[i] and trend_4h != -1:
                trend_vote = 1
            elif hma_slope < 0 and hma_fast_slope < 0 and price < hma_1h[i] and trend_4h != 1:
                trend_vote = -1
        
        # SIGNAL 2: MOMENTUM (MACD histogram + RSI direction)
        if macd_hist > 0 and rsi_val > 50 and rsi_val < 70:
            momentum_vote = 1
        elif macd_hist < 0 and rsi_val < 50 and rsi_val > 30:
            momentum_vote = -1
        
        # SIGNAL 3: MEAN REVERSION (BB position + Z-score + 4h RSI filter)
        if is_mr_regime or is_neutral_regime:
            # Long: price at lower BB, oversold RSI, negative Z-score
            if bb_position < 0.2 and rsi_val < RSI_OVERSOLD and zscore_val < -ZSCORE_EXTREME and trend_4h != -1:
                mr_vote = 1
            # Short: price at upper BB, overbought RSI, positive Z-score
            elif bb_position > 0.8 and rsi_val > RSI_OVERBOUGHT and zscore_val > ZSCORE_EXTREME and trend_4h != 1:
                mr_vote = -1
        
        # Count votes
        total_vote = trend_vote + momentum_vote + mr_vote
        vote_agreement = abs(total_vote)
        
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
            
            # Stoploss check (2.0*ATR)
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
                    signals[i] = SIZE_SINGLE / 2
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
                    signals[i] = -SIZE_SINGLE / 2
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
            
            # Check if we should reverse position
            if prev_side == 1 and total_vote <= -2:
                signals[i] = -SIZE_SINGLE
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                continue
            elif prev_side == -1 and total_vote >= 2:
                signals[i] = SIZE_SINGLE
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                continue
            
            # Hold position if no exit/reversal triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== ENTRY LOGIC - ENSEMBLE VOTING ==========
        
        # Need at least 2 signals agreeing for entry
        if vote_agreement >= 2:
            if total_vote >= 2:
                # Long entry - size based on agreement level
                if vote_agreement == 3:
                    size = SIZE_TRIPLE
                else:
                    size = SIZE_DOUBLE
                
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            
            elif total_vote <= -2:
                # Short entry - size based on agreement level
                if vote_agreement == 3:
                    size = SIZE_TRIPLE
                else:
                    size = SIZE_DOUBLE
                
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
    
    return signals