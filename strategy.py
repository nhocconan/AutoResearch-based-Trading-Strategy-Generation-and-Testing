#!/usr/bin/env python3
"""
Experiment #040: 1h Primary + 4h/12h HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 1h strategies failed (Sharpe=0.000) because entry conditions were
TOO STRICT (CRSI<15, session filters, volume filters all together = no trades).

This strategy uses the PROVEN pattern from current best (mtf_hma_rsi_zscore_v1 Sharpe=5.4):
1. 4h/12h HMA for TREND DIRECTION (only trade with HTF trend)
2. 1h RSI for ENTRY TIMING (pullback within trend, not extreme values)
3. Minimal additional filters (volume > 0.7x avg, not 1.5x)
4. NO session filter (was killing trades in #038)
5. LOOSE RSI thresholds: <40 for long, >60 for short (not <15/>85)

Why this should work:
- 4h HMA bullish + 1h RSI<40 = buy the dip in uptrend (common, generates trades)
- 4h HMA bearish + 1h RSI>60 = sell the rally in downtrend (common, generates trades)
- Targets 40-80 trades/year on 1h (Rule 10 compliant)
- Position size 0.25 (conservative for lower TF)

Key difference from failed #030, #035, #038:
- RSI(14) instead of CRSI (CRSI extremes too rare)
- RSI<40/>60 instead of <15/>85 (much more frequent signals)
- No session filter (8-20 UTC was filtering out 60% of bars)
- Volume filter: >0.7x 20-bar avg (not >1.5x)
- HTF: 4h HMA only (not 4h+12h+1d confluence = too strict)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for macro bias (secondary filter)
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_sma_20 = calculate_sma(volume, period=20)
    
    # 1h HMA for local trend confirmation
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        if np.isnan(hma_1h[i]):
            continue
        if atr_14[i] == 0 or vol_sma_20[i] == 0:
            continue
        
        # === HTF TREND DIRECTION (4h HMA) ===
        # 4h HMA slope (compare to 3 bars ago)
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 12H MACRO BIAS (secondary confirmation) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i] if not np.isnan(hma_12h_aligned[i]) else True
        price_below_hma_12h = close[i] < hma_12h_aligned[i] if not np.isnan(hma_12h_aligned[i]) else True
        
        # === 1H RSI PULLBACK (entry timing) ===
        # LOOSE thresholds to ensure trades generate
        rsi_oversold = rsi_14[i] < 40  # Was <15 in failed strategies
        rsi_overbought = rsi_14[i] > 60  # Was >85 in failed strategies
        rsi_neutral_low = rsi_14[i] < 45
        rsi_neutral_high = rsi_14[i] > 55
        
        # === VOLUME FILTER (light, not strict) ===
        volume_ok = volume[i] > 0.7 * vol_sma_20[i]  # Was >1.5x in failed strategies
        
        # === 1H HMA LOCAL TREND ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-5] if i >= 5 else False
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-5] if i >= 5 else False
        
        # === ENTRY LOGIC (LOOSE enough to generate 40-80 trades/year) ===
        new_signal = 0.0
        
        # LONG: 4h bullish trend + 1h RSI pullback + volume OK
        # Only require 4h HMA slope OR price above 4h HMA (not both = too strict)
        if (hma_4h_slope_bull or price_above_hma_4h):
            if rsi_oversold and volume_ok:
                # Bonus: 12h confirmation makes it stronger but not required
                if price_above_hma_12h or hma_1h_slope_bull:
                    new_signal = POSITION_SIZE
        
        # SHORT: 4h bearish trend + 1h RSI rally + volume OK
        if (hma_4h_slope_bear or price_below_hma_4h):
            if rsi_overbought and volume_ok:
                # Bonus: 12h confirmation makes it stronger but not required
                if price_below_hma_12h or hma_1h_slope_bear:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position and no new signal, hold (don't flip-flop)
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
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