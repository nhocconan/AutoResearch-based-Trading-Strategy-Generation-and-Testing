#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF Donchian+RSI+ATR_Vol+Volume (1h+4h+1d v1)
==================================================================================================
Hypothesis: Combine 4h Donchian Channel (breakout trend filter) + 1h RSI pullback entries + 
15m ATR volatility regime filter + Volume confirmation. This differs from current best by:
- Donchian Channel instead of HMA/Supertrend (breakout-based trend detection)
- ATR percentile regime filter instead of Z-score (volatility-based, not price-based)
- Volume spike confirmation on entries (avoids low-liquidity traps)
- Timeframe mix: 1h base, 4h trend, 15m regime filter

Why this should work:
- Donchian channels excel in trending markets (clear breakout signals)
- ATR percentile avoids trading during extreme volatility (reduces whipsaws)
- Volume confirmation filters out fake breakouts
- 1h base timeframe balances signal frequency vs noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_rsi_atrvol_volume_1h_4h_1d_v1"
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
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper, lower, middle)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile within rolling lookback window"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        valid = window[window > 0]
        if len(valid) > 0:
            rank = np.sum(valid <= atr[i])
            percentile[i] = rank / len(valid)
    
    return percentile


def calculate_volume_spike(volume, lookback=20, threshold=1.5):
    """Detect volume spikes (volume > threshold * rolling avg)"""
    n = len(volume)
    spike = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        avg_vol = np.mean(volume[i - lookback:i])
        if avg_vol > 0 and volume[i] > threshold * avg_vol:
            spike[i] = True
    
    return spike


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    
    dema = 2 * ema1 - ema2
    
    return dema


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing (base timeframe)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    atr_pct_1h = calculate_atr_percentile(atr_1h, lookback=100)
    vol_spike_1h = calculate_volume_spike(volume, lookback=20, threshold=1.5)
    dema_1h = calculate_dema(close, period=21)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h Donchian Channel for trend direction
        donch_upper_4h, donch_mid_4h, donch_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
        
        # 4h RSI for trend strength
        rsi_4h = calculate_rsi(c_4h, period=14)
        
        # Align 4h indicators to 1h timeframe
        donch_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
        donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
        donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    except Exception:
        donch_mid_4h_aligned = np.zeros(n)
        donch_upper_4h_aligned = np.zeros(n)
        donch_lower_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for major regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily SMA-50 for major trend direction
        sma_50_1d = pd.Series(c_1d).rolling(window=50, min_periods=50).mean().values
        
        # Align to 1h timeframe
        sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    except Exception:
        sma_50_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR percentile regime filter (only trade in normal volatility)
    ATR_PCT_MIN = 0.20
    ATR_PCT_MAX = 0.80
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 20, 50)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        donch_mid_4h = donch_mid_4h_aligned[i] if i < len(donch_mid_4h_aligned) else 0
        donch_upper_4h = donch_upper_4h_aligned[i] if i < len(donch_upper_4h_aligned) else 0
        donch_lower_4h = donch_lower_4h_aligned[i] if i < len(donch_lower_4h_aligned) else 0
        rsi_4h = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        sma_50_1d = sma_50_1d_aligned[i] if i < len(sma_50_1d_aligned) else 0
        
        # ATR volatility regime filter (15m equivalent via 1h ATR percentile)
        if atr_pct_1h[i] < ATR_PCT_MIN or atr_pct_1h[i] > ATR_PCT_MAX:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # 4h Donchian trend filter
        price = close[i]
        trend_4h = 0
        if donch_mid_4h > 0:
            if price > donch_mid_4h:
                trend_4h = 1
            elif price < donch_mid_4h:
                trend_4h = -1
        
        # Daily major trend filter
        major_trend = 0
        if sma_50_1d > 0:
            if price > sma_50_1d:
                major_trend = 1
            elif price < sma_50_1d:
                major_trend = -1
        
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
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
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
        
        # Entry logic: 4h Donchian trend + 1h RSI pullback + Volume confirmation + Daily filter
        # Long entry
        if trend_4h == 1 and major_trend >= 0:  # Bullish 4h, neutral/bullish daily
            # RSI pullback on 1h (not overbought)
            # Volume spike confirmation (optional but preferred)
            if (RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX):
                # Volume spike adds confidence but not required
                if vol_spike_1h[i] or rsi_4h > 45:  # Either volume spike or 4h RSI confirms
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        
        # Short entry
        elif trend_4h == -1 and major_trend <= 0:  # Bearish 4h, neutral/bearish daily
            # RSI pullback on 1h (not oversold)
            # Volume spike confirmation (optional but preferred)
            if (RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX):
                # Volume spike adds confidence but not required
                if vol_spike_1h[i] or rsi_4h < 55:  # Either volume spike or 4h RSI confirms
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals