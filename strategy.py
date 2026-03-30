#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian + Williams%R + Volume + 1d SMA200 Trend

HYPOTHESIS: Donchian(20) breakouts capture institutional momentum moves.
Williams %R filter (> -50 for longs, < -50 for shorts) ensures we're not
buying/selling into extreme readings that typically reverse.
Volume confirmation filters out false breakouts.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Breakout above Donchian high with Williams%R > -50 confirms bullish momentum
- Bear: Breakout below Donchian low with Williams%R < -50 confirms bearish momentum
- Range: Choppy market = no breakouts = no trades = no losses

KEY INSIGHT: Stacking ONE strong signal type (Donchian breakout) + volume +
trend filter + momentum filter = tight entries = fewer trades = less fee drag.
The Williams %R filter reduces entries by ~50% vs pure Donchian.

TARGET: 100-200 total trades over 4 years (25-50/year on 4h).
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_wr_vol_sma200_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # === Local 4h indicators ===
    # Williams %R(14) - momentum confirmation
    willr_14 = calculate_williams_r(high, low, close, period=14)
    
    # Donchian(20) channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: current volume vs 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 250  # Need 200 for SMA200 + 20 for Donchian + 14 for Williams%R
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if SMA200 not aligned
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        price_below_1d_sma = close[i] < sma_200_aligned[i]
        
        # === MOMENTUM (Williams %R) ===
        willr_momentum = willr_14[i]
        is_bullish_momentum = (not np.isnan(willr_momentum)) and (willr_momentum > -50)
        is_bearish_momentum = (not np.isnan(willr_momentum)) and (willr_momentum < -50)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (use PREVIOUS bar's Donchian to avoid look-ahead) ===
        prev_donch_high = donchian_high[i - 1] if i > 0 else 0
        prev_donch_low = donchian_low[i - 1] if i > 0 else float('inf')
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + volume + momentum + trend ===
            # Williams %R > -50 means not overbought, room for more upside
            if price_above_1d_sma and is_bullish_momentum and vol_spike:
                if close[i] > prev_donch_high:
                    desired_signal = SIZE
            
            # === SHORT: Breakout below Donchian low + volume + momentum + trend ===
            # Williams %R < -50 means oversold, more downside likely
            if price_below_1d_sma and is_bearish_momentum and vol_spike:
                if close[i] < prev_donch_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals