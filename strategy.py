#!/usr/bin/env python3
"""
Experiment #530: 1h Primary + 4h/12h HTF — Regime-Adaptive HMA + RSI + Session

Hypothesis: After 475 failed strategies (mostly complex volspike/choppiness combos),
try a REGIME-ADAPTIVE approach that switches logic based on market state.

Key insights from failures:
- Complex multi-condition entries = 0 trades or negative Sharpe
- Volspike strategies failed 20+ times - abandon this approach
- Choppiness Index alone doesn't work - need simpler regime detection
- 1h timeframe needs strict filters to avoid fee drag (>100 trades/yr = fail)

This strategy uses:
1. 12h HMA(21) for MAJOR trend direction (slow, reliable)
2. 4h HMA(16/48) for INTERMEDIATE trend confirmation
3. 1h RSI(3) for entry timing (Connors-style pullback entries)
4. UTC session filter (8-20 only - highest liquidity, avoids Asia chop)
5. Volume filter (>0.8x 20-bar avg) to confirm participation
6. ATR(14) 2.5x trailing stop for risk management
7. Asymmetric sizing: 0.25 in range regime, 0.35 in trend regime

Why this might work:
- 12h trend filter prevents counter-trend trades (major failure mode)
- 4h intermediate filter adds confluence without over-filtering
- RSI(3) is fast enough to catch pullbacks in trending markets
- Session filter reduces trades by ~40% (avoids low-liquidity hours)
- Volume filter confirms real participation, not fake breakouts
- Asymmetric sizing reduces exposure in choppy conditions

Position sizing: 0.25-0.35 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-60 trades/year on 1h (optimal fee/trade ratio)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_hma_rsi_session_4h12h_v1"
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
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR over period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    
    # CHOP formula
    chop = 100.0 * np.log10((atr_sum + 1e-10) / (hh - ll + 1e-10)) / np.log10(period)
    
    return chop.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF HMA for major trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 4h HTF HMA for intermediate trend
    hma_4h_16 = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, period=48)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_3 = calculate_rsi(close, 3)  # Fast RSI for entry timing
    rsi_14 = calculate_rsi(close, 14)  # Standard RSI for regime
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.35  # Larger size in trending regime
    SIZE_RANGE = 0.25  # Smaller size in ranging regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        if np.isnan(rsi_3[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === REGIME DETECTION ===
        # Choppiness Index: >55 = range, <45 = trend
        is_trending = chop_14[i] < 45.0
        is_ranging = chop_14[i] > 55.0
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        hma_slope_bull_12h = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_slope_bear_12h = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H INTERMEDIATE TREND (confluence filter) ===
        bull_regime_4h = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        bear_regime_4h = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        position_size = SIZE_RANGE if is_ranging else SIZE_TREND
        
        # LONG ENTRIES
        if bull_regime_12h and bull_regime_4h:
            # Trending regime: enter on RSI(3) pullback
            if is_trending and rsi_3[i] < 30.0 and in_session and volume_ok:
                new_signal = position_size
            # Ranging regime: enter on deeper RSI(3) oversold
            elif is_ranging and rsi_3[i] < 20.0 and in_session and volume_ok:
                new_signal = position_size
            # Strong trend confirmation: RSI(14) also supportive
            elif is_trending and rsi_3[i] < 40.0 and rsi_14[i] < 60.0 and in_session:
                new_signal = position_size * 0.9
        
        # SHORT ENTRIES
        if new_signal == 0.0 and bear_regime_12h and bear_regime_4h:
            # Trending regime: enter on RSI(3) bounce
            if is_trending and rsi_3[i] > 70.0 and in_session and volume_ok:
                new_signal = -position_size
            # Ranging regime: enter on deeper RSI(3) overbought
            elif is_ranging and rsi_3[i] > 80.0 and in_session and volume_ok:
                new_signal = -position_size
            # Strong trend confirmation: RSI(14) also supportive
            elif is_trending and rsi_3[i] > 60.0 and rsi_14[i] > 40.0 and in_session:
                new_signal = -position_size * 0.9
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme RSI) ===
        # Exit long on regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_12h and bear_regime_4h:
                new_signal = 0.0
            elif rsi_3[i] > 85.0:  # Extreme overbought on fast RSI
                new_signal = 0.0
        
        # Exit short on regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_12h and bull_regime_4h:
                new_signal = 0.0
            elif rsi_3[i] < 15.0:  # Extreme oversold on fast RSI
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
                # Flip position
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