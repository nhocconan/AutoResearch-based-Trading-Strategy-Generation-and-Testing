#!/usr/bin/env python3
"""
Experiment #634: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Extremes + Volume Filter

Hypothesis: Previous #624 had too many confluence requirements (12h slope + price vs HMA + 
4h cross + 4h slope + RSI zone + Donchian) causing poor risk-adjusted returns despite +39.9% 
return. This version simplifies to: 1d HMA trend bias + 4h RSI extremes + volume confirmation.

Key changes from #624:
1. Use 1d HTF instead of 12h (more stable trend signal, less noise)
2. RSI extremes (<35/>65) instead of pullback zones (clearer signals)
3. Volume spike confirmation (1.5x avg) to filter false breakouts
4. Asymmetric stoploss: 2.0*ATR for longs, 2.5*ATR for shorts (bear market bias)
5. Remove Donchian requirement (was causing missed entries)
6. Add trend strength filter (HMA slope > threshold) to avoid chop

Why this might beat Sharpe=0.520:
- Simpler logic = more consistent execution across all 3 symbols
- 1d trend filter captures major moves without whipsaw
- RSI extremes have proven mean-reversion edge in all regimes
- Volume filter reduces false signals (critical for 4h TF)
- Asymmetric stops account for 2025 bear market reality
- Target 25-40 trades/year on 4h (optimal for fee drag)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-40 trades/year on 4h (per Rule 10)
Stoploss: 2.0*ATR long / 2.5*ATR short (asymmetric for bear bias)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values > threshold

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_spike = calculate_volume_spike(volume, 20, 1.5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 5 bars for stability) ===
        hma_1d_slope = 0.0
        if i >= 5:
            hma_1d_slope = (hma_1d_aligned[i] - hma_1d_aligned[i-5]) / hma_1d_aligned[i-5]
        
        trend_bull = hma_1d_slope > 0.002  # >0.2% over 5 days
        trend_bear = hma_1d_slope < -0.002  # <-0.2% over 5 days
        trend_neutral = not trend_bull and not trend_bear
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H HMA TREND CONFIRMATION ===
        hma_4h_slope = 0.0
        if i >= 3:
            hma_4h_slope = (hma_4h[i] - hma_4h[i-3]) / hma_4h[i-3]
        
        hma_4h_bull = hma_4h_slope > 0.001
        hma_4h_bear = hma_4h_slope < -0.001
        
        # === RSI EXTREMES (simpler than pullback zones) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d bull trend + 4h confirmation + RSI oversold + volume ---
        # Condition 1: 1d HMA sloping up OR price above 1d HMA
        # Condition 2: 4h HMA sloping up (momentum confirmation)
        # Condition 3: RSI oversold (<35) for pullback entry
        # Condition 4: Volume spike confirms interest
        if (trend_bull or price_above_hma_1d):
            if hma_4h_bull:
                if rsi_oversold:
                    if volume_confirmed:
                        new_signal = POSITION_SIZE
                    else:
                        # Allow entry without volume if RSI very oversold
                        if rsi_14[i] < 25.0:
                            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 1d bear trend + 4h confirmation + RSI overbought + volume ---
        # Condition 1: 1d HMA sloping down OR price below 1d HMA
        # Condition 2: 4h HMA sloping down (momentum confirmation)
        # Condition 3: RSI overbought (>65) for bounce entry
        # Condition 4: Volume spike confirms interest
        elif (trend_bear or price_below_hma_1d):
            if hma_4h_bear:
                if rsi_overbought:
                    if volume_confirmed:
                        new_signal = -POSITION_SIZE
                    else:
                        # Allow entry without volume if RSI very overbought
                        if rsi_14[i] > 75.0:
                            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (asymmetric: tighter for shorts in bear market) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]  # 2.0*ATR for longs
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]  # 2.5*ATR for shorts
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if trend_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if trend_bull and price_above_hma_1d:
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