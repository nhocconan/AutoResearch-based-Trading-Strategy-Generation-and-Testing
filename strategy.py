#!/usr/bin/env python3
"""
Experiment #124: 4h Primary + 12h/1d HTF — Keltner Breakout with ADX Trend Filter

Hypothesis: Recent Donchian-based strategies (#114, #117, #123) failed because Donchian
channels are too wide and slow to react. Keltner Channels (ATR-based) adapt to volatility
and provide cleaner breakout signals. Combined with ADX trend filter and HTF bias:

1) 12h HMA(21) for macro trend bias — only trade breakouts in trend direction
2) 1d HMA(50) for secondary confirmation — adds robustness
3) Keltner Channel(20, 2.0*ATR) breakout — volatility-adaptive entry
4) ADX(14) > 25 filter — ensures we only trade in trending conditions (not chop)
5) ATR(14) trailing stop at 2.5x — locks profits, limits drawdown
6) Exit on opposite Keltner break OR HTF trend reversal

Why this should beat Donchian:
- Keltner uses ATR, adapts to volatility regime changes
- ADX filter prevents entries during chop (where Donchian whipsaws)
- Dual HTF (12h + 1d) provides stronger trend confirmation than single HTF
- 4h naturally produces 30-50 trades/year (optimal fee/trade ratio)

Position size: 0.25 base, 0.30 with strong confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_keltner_adx_hma_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_keltner(high, low, close, period=20, atr_mult=2.0):
    """Calculate Keltner Channel (EMA + ATR bands)."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, period=14)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, lower, ema

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Where both are positive, keep the larger one
    both_positive = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[both_positive & (plus_dm < minus_dm)] = 0
    minus_dm[both_positive & (minus_dm < plus_dm)] = 0
    
    # Smooth with Wilder's method (EMA with span=period)
    atr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr_smooth + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr_smooth + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for macro trend
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1d HMA for secondary confirmation
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    keltner_upper, keltner_lower, keltner_mid = calculate_keltner(high, low, close, period=20, atr_mult=2.0)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            continue
        if np.isnan(adx_14[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === HTF TREND BIAS (12h + 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias: both 12h and 1d agree
        strong_bullish = price_above_hma_12h and price_above_hma_1d
        strong_bearish = price_below_hma_12h and price_below_hma_1d
        
        # === 4h TREND FILTER ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === ADX TREND STRENGTH ===
        adx_trending = adx_14[i] > 25.0
        adx_strong = adx_14[i] > 30.0
        
        # === KELTNER BREAKOUT ===
        prev_upper = keltner_upper[i-1] if i > 0 else keltner_upper[i]
        prev_lower = keltner_lower[i-1] if i > 0 else keltner_lower[i]
        
        breakout_long = close[i] > prev_upper
        breakout_short = close[i] < prev_lower
        
        # === RSI CONFIRMATION (avoid extreme overbought/oversold entries) ===
        rsi_ok_long = rsi_14[i] < 70.0  # Not overbought
        rsi_ok_short = rsi_14[i] > 30.0  # Not oversold
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: HTF bullish + 4h trend up + ADX trending + Keltner breakout + RSI ok
        if strong_bullish or (price_above_hma_12h and hma_4h_bullish):
            if hma_4h_bullish and adx_trending and breakout_long and rsi_ok_long:
                if adx_strong and strong_bullish:
                    new_signal = POSITION_SIZE_MAX
                else:
                    new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Require: HTF bearish + 4h trend down + ADX trending + Keltner breakout + RSI ok
        if strong_bearish or (price_below_hma_12h and hma_4h_bearish):
            if hma_4h_bearish and adx_trending and breakout_short and rsi_ok_short:
                if adx_strong and strong_bearish:
                    new_signal = -POSITION_SIZE_MAX
                else:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold long if still above Keltner mid and HTF trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if close[i] > keltner_mid[i] and price_above_hma_12h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if close[i] < keltner_mid[i] and price_below_hma_12h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_12h or (hma_4h_bearish and adx_trending):
                new_signal = 0.0
            # Exit on opposite Keltner break
            if breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_12h or (hma_4h_bullish and adx_trending):
                new_signal = 0.0
            # Exit on opposite Keltner break
            if breakout_long:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals