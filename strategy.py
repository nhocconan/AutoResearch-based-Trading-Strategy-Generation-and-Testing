#!/usr/bin/env python3
"""
Experiment #022: 1d Donchian Trend + 12h RSI Extremes + Choppiness Regime

HYPOTHESIS: Combining two proven patterns for strict 12h entries:
1. 1d Donchian breakout for HTF trend direction (filters 75% of noise)
2. 12h RSI extremes for mean-reversion entry (tight, infrequent signals)
3. Choppiness Index regime (avoid range markets)
4. Volume spike confirmation

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: 1d Donchian break + RSI<25 + CHOP<40 → long (catch support bounces)
- Bear: 1d Donchian break + RSI>75 + CHOP<40 → short (resistance rejections)
- Range: CHOP>50 → no entries (prevents whipsaws in 2022 crash)
- HTF direction prevents fighting the trend

KEY INSIGHT: RSI extremes are rare at 12h (only 4-6 per year max).
Combined with 1d Donchian + CHOP filter → targeting 60-120 total trades over 4 years.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_rsi_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(prices, period=14):
    """RSI with min_periods"""
    delta = np.diff(prices, prepend=prices[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (no trend)
    CHOP < 38.2 = trending
    Values in between = choppy transition
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Sum of true range over period
        period_sum = 0
        for j in range(i - period + 1, i + 1):
            period_sum += high[j] - low[j]
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_hl = hh - ll
        
        if range_hl > 1e-10:
            # CHOP = 100 * log10(sum ATR) / log10(highest - lowest)
            chop[i] = 100 * np.log10(period_sum) / np.log10(range_hl)
    
    # Smooth with EMA
    chop_smooth = pd.Series(chop).ewm(span=5, min_periods=5, adjust=False).mean().values
    return chop_smooth

def calculate_volume_ratio(volume, period=20):
    """Volume spike detection"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume / np.where(vol_ma > 0, vol_ma, 1)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Donchian for trend direction ===
    # Use 20-period (4 weeks) on 1d for major trend
    donch_upper_1d, donch_mid_1d, donch_lower_1d = calculate_donchian(
        df_1d['high'].values, df_1d['low'].values, period=20
    )
    
    # 1d trend: price above 20d high = bull, below 20d low = bear
    htf_price_1d = df_1d['close'].values
    htf_bullish = htf_price_1d > donch_upper_1d
    htf_bearish = htf_price_1d < donch_lower_1d
    
    # Align to 12h
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # Also use 1d RSI for regime
    rsi_1d = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Donchian needs 20, ATR needs 14
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK ===
        # Only trade when CHOP < 50 (trending or transitioning)
        # When CHOP > 61.8, skip (ranging market)
        in_chop_range = chop_14[i] < 61.8
        
        # === HTF TREND ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else False
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else False
        
        # === 12h Donchian break ===
        donch_break_up = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1] if i > 0 else False
        donch_break_down = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1] if i > 0 else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === VOLUME SPIKE ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: 1d bullish + RSI oversold + volume + optional donch break
            # RSI < 30 is rare on 12h (~3-4 times per year), gives tight entries
            if htf_bull and rsi_oversold and vol_spike and in_chop_range:
                # HTF confirms uptrend, RSI gives mean-reversion entry
                desired_signal = SIZE
            elif htf_bull and donch_break_up and vol_spike and in_chop_range:
                # Donchian break as alternative (stronger momentum)
                desired_signal = SIZE
            
            # SHORT: 1d bearish + RSI overbought + volume + optional donch break
            if htf_bear and rsi_overbought and vol_spike and in_chop_range:
                desired_signal = -SIZE
            elif htf_bear and donch_break_down and vol_spike and in_chop_range:
                desired_signal = -SIZE
        
        # === STOPLOSS (3 ATR - wider for 12h) ===
        if in_position:
            if position_side > 0:
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                stop_price = trailing_high - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
                
                # Take profit at 2R
                profit_target = entry_price + 2.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = SIZE / 2  # Half position
                    in_position = True  # Keep tracking for full exit
            
            elif position_side < 0:
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                stop_price = trailing_low + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
                
                # Take profit at 2R
                profit_target = entry_price - 2.0 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = -SIZE / 2
                    in_position = True
        
        # === MINIMUM HOLD: 3 bars ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals