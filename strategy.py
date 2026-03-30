#!/usr/bin/env python3
"""
Experiment #022: Choppiness regime filter + Donchian breakout + Volume confirmation (4h)

HYPOTHESIS: Use choppiness index as regime filter to avoid ranging markets.
Only enter when CHOP < 45 (trending) + price breaks Donchian channel + volume spike.
This avoids whipsaws in sideways markets which destroyed many previous strategies.

WHY IT SHOULD WORK:
- Bull: CHOP<45 + price breaks upper Donchian + volume spike → strong trend continuation
- Bear: CHOP<45 + price breaks lower Donchian + volume spike → strong trend continuation  
- Range (CHOP>52): No entries, avoids 2022 crash whipsaws
- Choppiness Index is proven regime filter (top performers use it)

KEY INSIGHT from DB: Top 4h strategies use CHOP as meta-filter to avoid range markets.
Donchian(20-30) breakout + volume + CHOP filter = proven pattern.

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_vol_1d_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (Ehler's)
    CHOP > 61.8 = ranging (no trend)
    CHOP < 38.2 = strong trend
    Range: 38.2 - 61.8 = transitioning
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Sum of ATR over period
    atr_sum = np.zeros(n)
    for i in range(period - 1, n):
        atr_sum[i] = np.sum(high[i-period+1:i+1] - low[i-period+1:i+1])
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Range of the period
    period_range = hh - ll
    
    # Choppiness formula: 100 * log10(sum(ATR)) / log10(range)
    # Using ATR sum as proxy
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        if period_range[i] > 0 and atr_sum[i] > 0:
            chop[i] = 100 * (np.log10(atr_sum[i]) / np.log10(period_range[i]))
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, middle, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
                adx[i] = dx
    
    adx_smooth = pd.Series(adx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx_smooth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF: 1d close for trend direction ===
    htf_close = df_1d['close'].values
    htf_ma_20 = pd.Series(htf_close).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF to LTF (shift by 1 to avoid look-ahead)
    htf_bullish = (htf_close > htf_ma_20).astype(float)
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian Channel
    donchian_upper, donchian_middle, donchian_lower = calculate_donchian(high, low, period=20)
    
    # ADX for trend strength confirmation
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian needs 20, chop needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER (CHOPPINESS) ===
        # CHOP < 45 = trending, CHOP > 52 = ranging (no trade)
        trending = chop[i] < 45
        ranging = chop[i] > 52
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long: price breaks above upper channel
        above_upper = close[i] > donchian_upper[i]
        prev_below_upper = close[i-1] <= donchian_upper[i-1] if i > 0 else False
        donchian_long_breakout = above_upper and prev_below_upper
        
        # Short: price breaks below lower channel
        below_lower = close[i] < donchian_lower[i]
        prev_above_lower = close[i-1] >= donchian_lower[i-1] if i > 0 else False
        donchian_short_breakout = below_lower and prev_above_lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx[i] > 20
        
        # === HTF TREND ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Only trade in trending regime (CHOP < 45)
            if trending:
                # LONG: Donchian breakout + volume spike + HTF bull or neutral
                if donchian_long_breakout and (vol_spike or strong_trend):
                    if htf_bull:  # HTF confirms uptrend
                        desired_signal = SIZE
                
                # SHORT: Donchian breakout down + volume spike + HTF bear
                elif donchian_short_breakout and (vol_spike or strong_trend):
                    if not htf_bull:  # HTF confirms downtrend
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if chop regime ends (price enters range)
                if chop[i] > 55:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if not htf_bull and chop[i] > 45:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if chop regime ends
                if chop[i] > 55:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull and chop[i] > 45:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
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