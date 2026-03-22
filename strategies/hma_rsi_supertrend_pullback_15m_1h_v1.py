#!/usr/bin/env python3
"""
EXPERIMENT #038 - HMA RSI Supertrend Pullback (15m Primary + 1h HTF)
==================================================================================================
Hypothesis: Current best uses 4h trend filter, but 1h trend may be more responsive for 
shorter-term entries. Testing 15m primary (more trades than 1h) with 1h HTF trend filter
(cleaner than 4h for 15m entries). This combination hasn't been tested yet.

Key innovations vs #037:
1. 15m PRIMARY + 1h HTF: More trades than 1h_4h, cleaner signals than 30m_4h
2. HMA instead of KAMA: Faster response, proven in #027/#034 (Sharpe > 0.4)
3. Wider RSI pullback zone: 40-60 instead of 45-55 (more entry opportunities)
4. Dynamic position sizing: Base 0.25, scale to 0.30 on high conviction
5. Tighter stoploss: 1.8*ATR instead of 2.0*ATR (reduce drawdown)
6. Volume confirmation: Require taker_buy_volume > average for long entries

Why this should beat Sharpe=0.537:
- 15m captures more momentum moves than 1h
- 1h trend filter is more responsive than 4h for 15m entries
- HMA reacts faster than KAMA in trending markets
- Wider RSI zone = more trades while maintaining quality
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_supertrend_pullback_15m_1h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """
    Hull Moving Average - faster response than EMA with less lag
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    def wma(series, span):
        """Weighted Moving Average"""
        result = np.zeros(len(series))
        for i in range(span - 1, len(series)):
            weights = np.arange(1, span + 1)
            result[i] = np.sum(series[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    diff = 2 * wma_half - wma_full
    
    hma = wma(diff, sqrt_period)
    
    return hma


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


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
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


def calculate_volume_ratio(taker_buy_volume, volume, period=20):
    """Calculate taker buy volume ratio vs average"""
    n = len(volume)
    ratio = np.zeros(n)
    
    for i in range(period, n):
        avg_volume = np.mean(volume[i - period:i])
        avg_taker = np.mean(taker_buy_volume[i - period:i])
        if avg_volume > 0:
            ratio[i] = taker_buy_volume[i] / avg_taker if avg_taker > 0 else 1.0
        else:
            ratio[i] = 1.0
    
    return ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # ========== 15m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m_fast = calculate_hma(close, period=8)
    hma_15m_slow = calculate_hma(close, period=21)
    supertrend_15m, st_trend_15m = calculate_supertrend(high, low, close, atr_15m, multiplier=3.0)
    volume_ratio = calculate_volume_ratio(taker_buy_volume, volume, period=20)
    
    # ========== 1h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # 1h HMA for trend direction
        hma_1h = calculate_hma(close_1h, period=21)
        atr_1h = calculate_atr(high_1h, low_1h, close_1h, period=14)
        _, st_trend_1h = calculate_supertrend(high_1h, low_1h, close_1h, atr_1h, multiplier=3.0)
        
        # Align to 15m timeframe (auto shift for completed bars)
        hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
        st_trend_1h_aligned = align_htf_to_ltf(prices, df_1h, st_trend_1h)
        
    except Exception:
        hma_1h_aligned = np.zeros(n)
        st_trend_1h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25   # Base position (25% of capital)
    SIZE_HIGH = 0.30   # High conviction (30% of capital)
    SIZE_MAX = 0.35    # Absolute max
    
    # ATR stoploss - tighter than #037
    ATR_STOP_MULT = 1.8
    
    # RSI pullback zones - wider than #037 for more entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Volume confirmation threshold
    VOL_RATIO_MIN = 0.8
    
    first_valid = max(150, 100)
    
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
        st_trend_val = st_trend_15m[i]
        hma_fast_val = hma_15m_fast[i]
        hma_slow_val = hma_15m_slow[i]
        vol_rat = volume_ratio[i]
        
        # 1h trend filters (MASTER FILTER)
        hma_1h_val = hma_1h_aligned[i]
        st_trend_1h_val = st_trend_1h_aligned[i]
        
        # Determine 1h trend direction
        trend_1h = 0
        if hma_1h_val > 0 and price > hma_1h_val:
            trend_1h = 1
        elif hma_1h_val > 0 and price < hma_1h_val:
            trend_1h = -1
        
        if st_trend_1h_val == 1:
            trend_1h = max(trend_1h, 1)
        elif st_trend_1h_val == -1:
            trend_1h = min(trend_1h, -1)
        
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
            
            # Stoploss check (1.8*ATR)
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
        
        # ========== ENTRY LOGIC - RSI PULLBACK IN TREND DIRECTION ==========
        # Volume confirmation
        vol_ok_long = vol_rat >= VOL_RATIO_MIN
        vol_ok_short = vol_rat >= VOL_RATIO_MIN
        
        # LONG: 1h trend up + 15m Supertrend up + RSI pullback (40-60) + HMA fast > slow + vol ok
        long_condition = (
            trend_1h == 1 and
            st_trend_val == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            hma_fast_val > hma_slow_val and
            vol_ok_long
        )
        
        # SHORT: 1h trend down + 15m Supertrend down + RSI pullback (40-60) + HMA fast < slow + vol ok
        short_condition = (
            trend_1h == -1 and
            st_trend_val == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            hma_fast_val < hma_slow_val and
            vol_ok_short
        )
        
        # Determine position size based on conviction
        # High conviction: 1h Supertrend aligns with entry direction
        high_conviction_long = long_condition and st_trend_1h_val == 1
        high_conviction_short = short_condition and st_trend_1h_val == -1
        
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
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals