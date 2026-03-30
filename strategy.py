#!/usr/bin/env python3
"""
Experiment #021: 12h Vortex + Donchian Breakout + Choppiness

HYPOTHESIS: Vortex Indicator (VI) measures directional movement strength.
Unlike RSI/MACD which measure oscillation, VI captures TREND DIRECTION better.
Combined with Donchian breakout and volume spike, this should work in ALL markets:
- Bull: VI+ > VI- + price above Donchian = strong longs
- Bear: VI- > VI+ + price below Donchian = strong shorts  
- Range: Choppiness > 61.8 = SKIP (avoids choppy whipsaws)
- Vol spike confirms breakout validity, reducing false signals

DB Analysis: "mtf_4h_chop_donchian_vol_regime_12h_v1" (107 tr, Sharpe 1.491) uses
similar structure. Vortex adds directional confirmation without adding complexity.

TARGET: 80-150 total trades over 4 years (20-37/year on 12h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vortex_donchian_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_vortex(high, low, close, period=14):
    """
    Vortex Indicator (VT)
    VI+ measures upward trend strength
    VI- measures downward trend strength
    VI+ > VI- → bullish, VI- > VI+ → bearish
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    vm_plus = np.zeros(n)
    vm_minus = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # Vortex movement (direction strength)
        vm_plus[i] = abs(high[i] - low[i-1])
        vm_minus[i] = abs(low[i] - high[i-1])
        
        # True range
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Sum over period
    period_range = period
    
    vi_plus = np.full(n, np.nan)
    vi_minus = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_vm_plus = np.sum(vm_plus[i - period + 1:i + 1])
        sum_vm_minus = np.sum(vm_minus[i - period + 1:i + 1])
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        
        if sum_tr > 0:
            vi_plus[i] = sum_vm_plus / sum_tr
            vi_minus[i] = sum_vm_minus / sum_tr
    
    return vi_plus, vi_minus

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 50 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    return upper, lower

def calculate_williams_r(high, low, close, period=14):
    """
    Williams %R - momentum oscillator for entry timing
    < -80 = oversold (good for long entry)
    > -20 = overbought (good for short entry)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    wr = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest:
            wr[i] = -100 * (highest - close[i]) / (highest - lowest)
    
    return wr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(21) for trend direction
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # === Local 12h indicators ===
    vi_plus, vi_minus = calculate_vortex(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    channel_up, channel_lo = calculate_donchian(high, low, period=20)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
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
    
    # ATR for stoploss calculation
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    warmup = 250  # 200 for channel + 14 for CHOP + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(channel_up[i]) or np.isnan(channel_lo[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 50
        
        # Skip choppy markets entirely
        if is_choppy:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === VORTEX DIRECTION ===
        vi_diff = vi_plus[i] - vi_minus[i]  # positive = bullish, negative = bearish
        vortex_bullish = vi_diff > 0
        vortex_bearish = vi_diff < 0
        
        # === HTF TREND: 1d EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === VOLUME CONFIRMATION (1.5x minimum) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (shift 1 = previous bar, not current) ===
        breakout_up = close[i] > channel_up[i]
        breakout_down = close[i] < channel_lo[i]
        
        # === WILLIAMS %R for momentum confirmation ===
        wr_val = williams_r[i]
        if np.isnan(wr_val):
            wr_bullish = False
            wr_bearish = False
        else:
            wr_bullish = wr_val < -70  # Oversold - good for longs
            wr_bearish = wr_val > -30  # Overbought - good for shorts
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + vortex bullish + breakout up + HTF up + volume ===
            if breakout_up and htf_trend_up and vol_spike and is_trending:
                if vortex_bullish:  # Vortex confirms upward momentum
                    desired_signal = SIZE
            
            # === SHORT: Trending + vortex bearish + breakout down + HTF down + volume ===
            if breakout_down and htf_trend_down and vol_spike and is_trending:
                if vortex_bearish:  # Vortex confirms downward momentum
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Long stop: entry - 2.5 ATR
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips or vortex turns bearish
                if htf_trend_down or vortex_bearish:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop: entry + 2.5 ATR
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips or vortex turns bullish
                if htf_trend_up or vortex_bullish:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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