#!/usr/bin/env python3
"""
Experiment #022: Simple Donchian Breakout + Volume Confirmation + Choppiness Regime (4h)

HYPOTHESIS: The simplest possible strategy that works:
1. Donchian(20) breakout structure - proven price channel from DB
2. Volume spike confirmation - filters false breakouts
3. Choppiness Index regime - avoid whipsaws in range-bound markets

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above Donchian high + volume spike + CHOP<38 = strong momentum
- Bear: Price breaks below Donchian low + volume spike + CHOP>61 = strong bear trend
- Range: CHOP>61 = stay out (mean reversion mode)

KEY INSIGHT from DB: Best performers (Sharpe 1.3-1.8) use:
- Tight entry conditions (~75-300 train trades)
- Volume confirmation
- Price channel structure (Donchian)
- Regime filter (chop/ADX)
- Simple = fewer trades = less fee drag

TARGET: 80-200 total trades over 4 years (20-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_simple_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    - CHOP > 61.8 = ranging market (no trend, mean reversion)
    - CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0 and atr_sum > 0:
            chop[i] = 100 * np.log(atr_sum / range_sum) / np.log(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20 period breakout"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # === HTF: Donchian for trend direction ===
    htf_upper, htf_lower, htf_middle = calculate_donchian(
        df_12h['high'].values, df_12h['low'].values, period=20
    )
    htf_close = df_12h['close'].values
    htf_price_above = htf_close > htf_upper
    htf_price_below = htf_close < htf_lower
    htf_price_above_aligned = align_htf_to_ltf(prices, df_12h, htf_price_above.astype(float))
    htf_price_below_aligned = align_htf_to_ltf(prices, df_12h, htf_price_below.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 period)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    
    # Choppiness Index
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume ratio (20 period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 150  # Donchian=20, ATR=14, volume=20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === Regime check ===
        chop_trending = chop[i] < 38.2  # Trending
        chop_ranging = chop[i] > 61.8   # Ranging - stay out
        
        # === Volume confirmation ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HTF trend ===
        htf_bull = htf_price_above_aligned[i] > 0.5 if not np.isnan(htf_price_above_aligned[i]) else False
        htf_bear = htf_price_below_aligned[i] > 0.5 if not np.isnan(htf_price_below_aligned[i]) else False
        
        # === Entry conditions ===
        desired_signal = 0.0
        
        if not in_position:
            # Donchian breakout logic
            price_breaks_high = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
            price_breaks_low = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
            
            # LONG: Price breaks above Donchian + volume spike + trending regime + HTF bull
            if price_breaks_high and vol_spike and chop_trending:
                if htf_bull or not htf_bear:  # Bull or neutral HTF
                    desired_signal = SIZE
            
            # SHORT: Price breaks below Donchian + volume spike + ranging or strong bear regime
            elif price_breaks_low and vol_spike:
                if htf_bear or not htf_bull:  # Bear or neutral HTF
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position:
            if position_side > 0:
                # Long stoploss
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if price falls below Donchian middle (trend weakening)
                if close[i] < donchian_middle[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stoploss
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if price rises above Donchian middle (trend weakening)
                if close[i] > donchian_middle[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals