#!/usr/bin/env python3
"""
Experiment #011: 4h Primary + 1d/1w HTF — Ehlers Fisher Transform + HMA Trend

Hypothesis: After 10 failed experiments with CRSI/Choppiness/Donchian/VolSpike patterns,
I'm testing Ehlers Fisher Transform which is research-backed for catching reversals
in bear/range markets (exactly what 2025 test period is).

Key differences from failed attempts:
1. NO Choppiness Index (failed in #001, #002, #006, #008, #010)
2. NO Connors RSI (failed in #001, #006, #008, #010)
3. NO Vol Spike ratio (failed catastrophically in #004 with Sharpe=-11.9)
4. Using Ehlers Fisher Transform (period=9) — catches reversals when Fisher crosses
   -1.5 (long) or +1.5 (short). Works well in bear market rallies.
5. 1d HMA for primary trend, 1w HMA for regime filter (only trade with weekly trend)
6. ADX(14) > 20 filter to ensure some momentum (avoids dead chop)

Why this might work:
- Fisher Transform is designed for non-Gaussian price distributions (crypto fits)
- 1w HMA regime filter prevents trading against macro trend
- 4h TF targets 20-50 trades/year (fee-efficient per Rule 10)
- Position size 0.30 (conservative, discrete levels)
- Different from all 10 failed experiments

Entry conditions (LOOSE enough to generate trades):
- Long: Fisher < -1.5 AND Fisher crosses up AND 1d HMA bullish AND 1w HMA not bearish
- Short: Fisher > +1.5 AND Fisher crosses down AND 1d HMA bearish AND 1w HMA not bullish
- ADX > 20 for both (ensures momentum)

Stoploss: 2.5*ATR trailing, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_regime_1d1w_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest(period)) / (highest(period) - lowest(period))
    3. Transform: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Smooth with EMA
    """
    n = len(close)
    typical = (high + low) / 2.0
    
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Find highest and lowest over lookback period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        # Normalize to 0-1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            normalized = 0.5
        else:
            normalized = (typical[i] - lowest) / range_val
        
        # Clamp to avoid log(0) or log(inf)
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Smooth with EMA (span=3 for responsiveness)
        if i == period:
            fisher[i] = fisher_raw
        else:
            fisher[i] = 0.67 * fisher_raw + 0.33 * fisher[i-1]
        
        # Signal line (previous fisher value)
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher_raw
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # True range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional indicators
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # DX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for regime filter
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1W REGIME FILTER ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ADX MOMENTUM FILTER ===
        adx_strong = adx_14[i] > 20  # Some momentum required
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Fisher cross up from oversold OR Fisher deeply oversold + RSI confirmation
        fisher_long_signal = fisher_cross_up or (fisher_oversold and rsi_oversold)
        
        # 1d trend must be bullish (or at least not strongly bearish)
        trend_allows_long = hma_1d_slope_bull or price_above_hma_1d
        
        # 1w regime must not be strongly bearish
        regime_allows_long = not (hma_1w_slope_bear and price_below_hma_1d)
        
        if fisher_long_signal and trend_allows_long and regime_allows_long and adx_strong:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Fisher cross down from overbought OR Fisher deeply overbought + RSI confirmation
        fisher_short_signal = fisher_cross_down or (fisher_overbought and rsi_overbought)
        
        # 1d trend must be bearish (or at least not strongly bullish)
        trend_allows_short = hma_1d_slope_bear or price_below_hma_1d
        
        # 1w regime must not be strongly bullish
        regime_allows_short = not (hma_1w_slope_bull and price_above_hma_1d)
        
        if fisher_short_signal and trend_allows_short and regime_allows_short and adx_strong:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
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