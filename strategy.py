#!/usr/bin/env python3
"""
Experiment #025: 12h Weekly Structure + ATR Volatility Expansion

HYPOTHESIS: Weekly structural levels (Donchian + EMA) provide the primary
trend framework. ATR volatility expansion identifies high-probability 
momentum bursts. This combination captures institutional moves while
filtering noise through multiple confirmation layers.

WHY 12h:
- Weekly structure on 12h = natural 2-bar-per-day progression
- Target: 60-120 trades over 4 years (15-30/year) — tight but valid
- Avoids 4h overtrading (915 trades failed), 1h fee drag

WHY IT SHOULD WORK IN BULL + BEAR:
- Bull: price above weekly EMA200 + ATR expansion breakout = momentum long
- Bear: price below weekly EMA200 + ATR expansion breakdown = momentum short  
- Range: both sides fail chop filter → no trades

ENTRY SIGNALS (strict — must ALL align):
1. Weekly EMA200 trend direction (required)
2. ATR(14) expansion > 1.8x 20-period MA (momentum burst)
3. Volume confirmation > 1.5x MA (smart money)
4. Donchian(20) touch/break of the channel

EXIT SIGNALS:
- ATR-based stop: 2.5x ATR from entry
- Trail stop: HH/HL for longs, LH/LL for shorts
- TRIX flip exits after minimum 2 bars

TARGET: 60-120 total trades over 4 years, Sharpe > 0, DD < -35%
Signal size: 0.25 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_weekly_structure_atr_expansion_1w_v1"
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

def calculate_trix(close, period=9):
    """TRIX: triple smoothed EMA rate of change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    trix = np.full(n, 0.0)
    for i in range(period * 3, n):
        if ema3[i - 1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
    
    return trix

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper/lower bounds"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 for multi-timeframe trend
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Weekly EMA50 for shorter-term trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix_9 = calculate_trix(close, period=9)
    donchian_up, donchian_lo, donchian_mid = calculate_donchian(high, low, period=20)
    
    # ATR expansion ratio (current ATR vs 20-period MA of ATR)
    atr_ma20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_expansion = atr_14 / np.where(atr_ma20 > 0, atr_ma20, 1)
    
    # Volume ratio (current vs 20-period MA)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    entry_high = 0.0
    entry_low = float('inf')
    
    warmup = 300  # Need enough for EMA200 alignment + indicator warmup
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_200_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION ===
        trix_bullish = trix_9[i] > 0
        trix_bearish = trix_9[i] < 0
        
        # Weekly trend: EMA200 primary, EMA50 confirmation
        weekly_bull = close[i] > ema_200_aligned[i] and close[i] > ema_50_aligned[i]
        weekly_bear = close[i] < ema_200_aligned[i] and close[i] < ema_50_aligned[i]
        
        # === VOLATILITY EXPANSION ===
        atr_expanded = atr_expansion[i] > 1.8  # ATR burst
        atr_normal = atr_expansion[i] < 1.4     # Not already expanded (avoid late entries)
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.5
        
        # === DONCHIAN TOUCH (within 0.5% of channel) ===
        channel_touch_up = close[i] >= donchian_up[i - 1] * 0.998 if not np.isnan(donchian_up[i - 1]) else False
        channel_touch_down = close[i] <= donchian_lo[i - 1] * 1.002 if not np.isnan(donchian_lo[i - 1]) else False
        
        # === CHOPPINESS (optional filter for ranging) ===
        # Skip choppy markets (both conditions must fail for no-trade)
        in_range = (close[i] < donchian_up[i - 1] * 0.97 and close[i] > donchian_lo[i - 1] * 1.03) if not np.isnan(donchian_up[i - 1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # ALL conditions must align:
            # 1. Weekly trend bullish
            # 2. ATR expansion (momentum burst)
            # 3. Volume confirmation
            # 4. TRIX bullish
            if weekly_bull and trix_bullish:
                # Primary: Donchian upper breakout with all confirmations
                if channel_touch_up and vol_confirmed and atr_normal:
                    desired_signal = SIZE
                # Alternative: ATR expansion pullback to mid-channel
                elif atr_expanded and vol_confirmed and close[i] > donchian_mid[i - 1] if not np.isnan(donchian_mid[i - 1]) else False:
                    desired_signal = SIZE * 0.5  # Half size for pullback
            
            # === SHORT ENTRY ===
            if weekly_bear and trix_bearish:
                # Primary: Donchian lower breakdown with all confirmations
                if channel_touch_down and vol_confirmed and atr_normal:
                    desired_signal = -SIZE
                # Alternative: ATR expansion rally to mid-channel
                elif atr_expanded and vol_confirmed and close[i] < donchian_mid[i - 1] if not np.isnan(donchian_mid[i - 1]) else False:
                    desired_signal = -SIZE * 0.5
        
        # === EXIT LOGIC ===
        if in_position:
            bars_held = i - entry_bar
            
            # Update trailing highs/lows
            if high[i] > entry_high:
                entry_high = high[i]
            if low[i] < entry_low:
                entry_low = low[i]
            
            # === ATR STOPLOSS (2.5x from entry) ===
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Trailing stop: lock in profits if price pulls back from high
                elif close[i] < entry_high - 2.0 * atr_14[i]:
                    desired_signal = SIZE * 0.5  # Take partial profits
                # TRIX flip exit (after minimum hold)
                elif trix_bearish and bars_held >= 3:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            elif position_side < 0:
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Trailing stop: lock in profits if price pulls back from low
                elif close[i] > entry_low + 2.0 * atr_14[i]:
                    desired_signal = -SIZE * 0.5  # Take partial profits
                # TRIX flip exit (after minimum hold)
                elif trix_bullish and bars_held >= 3:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
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
                entry_high = high[i]
                entry_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals