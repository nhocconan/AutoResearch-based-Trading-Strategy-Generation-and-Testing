#!/usr/bin/env python3
"""
EXPERIMENT #068 - Donchian Channel Breakout with Volume Confirmation (1h + 4h)
==================================================================================================
Hypothesis: Current best uses Supertrend+MACD+RSI. Let me try a PURE breakout approach:

Key insights from failures:
- Ensemble voting creates conflicting signals
- Too many indicators = overfitting and crashes
- Regime detection alone doesn't work
- KAMA/Supertrend/HMA combinations are overused

New approach:
- Donchian Channel (20-period high/low) - pure price action breakout
- 4h Donchian for major trend direction (breakout above 20-period high = bullish regime)
- 1h Donchian for entry triggers (breakout with volume confirmation)
- Volume spike confirmation (volume > 1.5x SMA = real breakout, not fakeout)
- ATR-based stoploss (2.5*ATR) and takeprofit (2R then trail)
- Position sizing: discrete levels (0.0, ±0.25, ±0.35) based on 4h trend strength
- NO complex regime detection - let the Donchian channels define the regime

Why this should beat Sharpe=3.653:
- Donchian channels are proven trend-following tools (Turtle Trading)
- Volume confirmation filters false breakouts (major source of losses)
- Multi-timeframe alignment (4h trend + 1h entries) reduces whipsaws
- Simpler logic = fewer bugs and crashes
- Conservative position sizing (max 0.35) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_volume_breakout_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian_channels(high, low, period=20):
    """
    Donchian Channels: Upper = highest high, Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, middle, lower


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


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rsi = np.full(n, 50.0)
    mask = avg_loss > 0
    rsi[mask] = 100 - (100 / (1 + avg_gain[mask] / avg_loss[mask]))
    rsi[~mask] = 100
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # === 1h indicators (entry timing) ===
    donch_upper_1h, donch_mid_1h, donch_lower_1h = calculate_donchian_channels(high, low, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    rsi_1h = calculate_rsi(close, period=14)
    
    # === 4h indicators (trend regime) using mtf_data helper ===
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h Donchian channels
        donch_upper_4h, donch_mid_4h, donch_lower_4h = calculate_donchian_channels(h_4h, l_4h, period=20)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
        donch_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
        donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
        
    except Exception:
        # Fallback: use 1h data only if 4h not available
        donch_upper_4h_aligned = donch_upper_1h
        donch_mid_4h_aligned = donch_mid_1h
        donch_lower_4h_aligned = donch_lower_1h
    
    # === Position sizing (discrete levels) ===
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_EXIT = 0.0
    
    # === Entry thresholds ===
    VOLUME_SPIKE_MULT = 1.5
    ATR_STOP_MULT = 2.5
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # === Initialize tracking arrays ===
    signals = np.zeros(n)
    position_side = np.zeros(n, dtype=np.int8)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    first_valid = 250  # Need enough data for Donchian and ATR
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if np.isnan(donch_upper_4h_aligned[i]) or np.isnan(donch_lower_4h_aligned[i]):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        vol_ratio = volume[i] / volume_sma_1h[i] if volume_sma_1h[i] > 0 else 1.0
        
        # 4h trend regime
        donch_4h_range = donch_upper_4h_aligned[i] - donch_lower_4h_aligned[i]
        donch_4h_position = (price - donch_lower_4h_aligned[i]) / donch_4h_range if donch_4h_range > 0 else 0.5
        
        # 1h channel levels
        donch_upper = donch_upper_1h[i]
        donch_lower = donch_lower_1h[i]
        donch_mid = donch_mid_1h[i]
        
        # === Check existing positions (stoploss & take profit) ===
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R after TP triggered
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
                
                # Take profit (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R after TP triggered
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
            
            # Hold position - no new entry signal
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # === NEW ENTRY LOGIC ===
        
        # Determine 4h trend regime
        # Price in upper half of 4h Donchian = bullish regime
        # Price in lower half of 4h Donchian = bearish regime
        bullish_regime = donch_4h_position > 0.5
        bearish_regime = donch_4h_position < 0.5
        
        # LONG entry: 1h breakout above Donchian upper + volume spike + 4h bullish
        if bullish_regime:
            if price > donch_upper and vol_ratio >= VOLUME_SPIKE_MULT:
                # RSI filter - not extremely overbought
                if rsi_1h[i] < RSI_OVERBOUGHT:
                    # Strong trend if price well above 4h middle
                    if price > donch_mid_4h_aligned[i] * 1.02:
                        signals[i] = SIZE_STRONG
                    else:
                        signals[i] = SIZE_BASE
                    
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    continue
        
        # SHORT entry: 1h breakout below Donchian lower + volume spike + 4h bearish
        if bearish_regime:
            if price < donch_lower and vol_ratio >= VOLUME_SPIKE_MULT:
                # RSI filter - not extremely oversold
                if rsi_1h[i] > RSI_OVERSOLD:
                    # Strong trend if price well below 4h middle
                    if price < donch_mid_4h_aligned[i] * 0.98:
                        signals[i] = -SIZE_STRONG
                    else:
                        signals[i] = -SIZE_BASE
                    
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    continue
        
        # No entry signal
        signals[i] = 0.0
        position_side[i] = 0
        entry_price[i] = 0
        tp_triggered[i] = False
        highest_since_entry[i] = 0
        lowest_since_entry[i] = 0
    
    return signals