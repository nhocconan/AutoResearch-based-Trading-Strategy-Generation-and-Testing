#!/usr/bin/env python3
"""
EXPERIMENT #057 - ADX Regime + KAMA Trend + RSI Pullback + Volume (30m Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.563) uses 1h primary with BB regime detection.
This strategy improves by:
1. Using 30m primary timeframe → more trade opportunities than 1h
2. ADX for regime detection → better trend/chop identification than BB width
3. KAMA adaptive trend → proven in #045, #047 (Sharpe=0.49-0.53)
4. Volume confirmation → filters false breakouts
5. 4h HMA trend filter → proven MTF combination

Why this should beat current best (Sharpe=0.563):
- 30m captures more intraday moves than 1h (2x trade frequency)
- ADX > 25 = trending, ADX < 20 = chopping (more precise than BB percentile)
- KAMA adapts to volatility better than Supertrend
- Volume filter reduces false signals in low-liquidity periods
- Conservative sizing (0.25 base, 0.35 high) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "adx_kama_rsi_volume_30m_4h_v1"
timeframe = "30m"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise - fast in trends, slow in chop
    """
    n = len(close)
    if n < period + slow:
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
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    first_valid = period
    kama[first_valid] = close[first_valid]
    
    # Calculate KAMA
    for i in range(first_valid + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = chopping
    """
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        # Directional Movement
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    # Smooth DM and TR
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # First ADX calculation at period*2
    start_idx = period
    
    for i in range(start_idx, n):
        if i == start_idx:
            sum_plus_dm = np.sum(plus_dm[start_idx - period + 1:start_idx + 1])
            sum_minus_dm = np.sum(minus_dm[start_idx - period + 1:start_idx + 1])
            sum_tr = np.sum(tr[start_idx - period + 1:start_idx + 1])
        else:
            sum_plus_dm = sum_plus_dm - plus_dm[i - 1] + plus_dm[i]
            sum_minus_dm = sum_minus_dm - minus_dm[i - 1] + minus_dm[i]
            sum_tr = sum_tr - tr[i - 1] + tr[i]
        
        if sum_tr > 0:
            plus_di[i] = 100 * sum_plus_dm / sum_tr
            minus_di[i] = 100 * sum_minus_dm / sum_tr
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        # DX calculation
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx_start = start_idx + period
    if adx_start < n:
        adx[adx_start] = np.mean(dx[adx_start - period:adx_start])
        for i in range(adx_start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_volume_sma(volume, period=20):
    """Calculate Volume SMA for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    kama_30m = calculate_kama(close, period=10, fast=2, slow=30)
    adx_30m, plus_di_30m, minus_di_30m = calculate_adx(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(close_4h, period=10, fast=2, slow=30)
        
        # 4h ADX for trend strength confirmation
        adx_4h, _, _ = calculate_adx(high_4h, low_4h, close_4h, period=14)
        
        # 4h HMA for additional trend confirmation
        from mtf_data import align_htf_to_ltf
        hma_4h_raw = calculate_hma(close_4h, period=21)
        
        # Align to 30m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
        hma_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25   # Base position (25% of capital)
    SIZE_HIGH = 0.35   # High conviction (35% of capital)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI thresholds for pullback entries
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    
    # ADX regime thresholds
    ADX_TRENDING = 25   # Above = trending market
    ADX_CHOPPING = 20   # Below = chopping market
    
    # Volume confirmation threshold
    VOL_CONFIRM_MULT = 1.2  # Volume must be > 1.2x SMA for breakout confirmation
    
    first_valid = max(200, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_30m[i]) or atr_30m[i] == 0 or np.isnan(rsi_30m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_30m[i]
        rsi_val = rsi_30m[i]
        kama_val = kama_30m[i]
        adx_val = adx_30m[i]
        plus_di = plus_di_30m[i]
        minus_di = minus_di_30m[i]
        vol_ratio = volume[i] / vol_sma_30m[i] if vol_sma_30m[i] > 0 else 0
        
        # 4h trend filters
        kama_4h_val = kama_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        hma_4h_val = hma_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if kama_4h_val > 0 and hma_4h_val > 0:
            if price > kama_4h_val and price > hma_4h_val:
                trend_4h = 1
            elif price < kama_4h_val and price < hma_4h_val:
                trend_4h = -1
        
        # Determine 30m regime
        is_trending = adx_val > ADX_TRENDING
        is_chopping = adx_val < ADX_CHOPPING
        
        # Determine 30m trend direction (KAMA + DI)
        trend_30m = 0
        if price > kama_val and plus_di > minus_di:
            trend_30m = 1
        elif price < kama_val and minus_di > plus_di:
            trend_30m = -1
        
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
                    signals[i] = SIZE_BASE / 2  # Reduce to half
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
                    signals[i] = -SIZE_BASE / 2  # Reduce to half
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
        
        # ========== ENTRY LOGIC - REGIME-BASED ==========
        
        # Volume confirmation
        vol_confirmed = vol_ratio >= VOL_CONFIRM_MULT
        
        # LONG conditions
        # Trending regime: KAMA trend + ADX confirming + volume
        long_trending = (is_trending and trend_30m == 1 and trend_4h != -1 and vol_confirmed)
        
        # Chopping regime: RSI oversold pullback + 4h trend not bearish
        long_pullback = (is_chopping and rsi_val < RSI_OVERSOLD and trend_4h != -1)
        
        # High conviction: trending + 4h aligns + volume spike
        high_conviction_long = (is_trending and trend_30m == 1 and trend_4h == 1 and vol_ratio > 1.5)
        
        # SHORT conditions
        short_trending = (is_trending and trend_30m == -1 and trend_4h != 1 and vol_confirmed)
        short_pullback = (is_chopping and rsi_val > RSI_OVERBOUGHT and trend_4h != 1)
        
        high_conviction_short = (is_trending and trend_30m == -1 and trend_4h == -1 and vol_ratio > 1.5)
        
        long_condition = long_trending or long_pullback
        short_condition = short_trending or short_pullback
        
        if long_condition:
            size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
    
    return signals


def calculate_hma(close, period=21):
    """
    Hull Moving Average - faster response than EMA with less lag
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    
    # WMA helper
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