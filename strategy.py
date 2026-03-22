#!/usr/bin/env python3
"""
Experiment #515: 1h Primary + 4h/1d HTF — Regime-Based Momentum with Session Filter

Hypothesis: After 461 failed strategies (mostly complex confluence = 0 trades), 
try a SIMPLER approach that actually generates trades:

1. 1D HMA(21) for MAJOR trend regime (bull/bear)
2. 4H RSI(14) for intermediate momentum confirmation  
3. 1H price breakout (20-bar high/low) for entry timing
4. Session filter (8-20 UTC) to avoid Asian session noise
5. Volume confirmation (>0.8x avg) for conviction

Why this might work:
- SIMPLER conditions = MORE trades (critical: need >=30/symbol on train)
- Session filter reduces false signals during low-liquidity hours
- 1h TF with HTF direction = ~40-60 trades/year (optimal fee/trade ratio)
- Different from 15+ failed vol-spike strategies

Position sizing: 0.25 (discrete, max 0.40)
Stoploss: 2.5 * ATR(14) trailing
Target: 40-60 trades/year, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_momentum_session_4h1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 4h HTF indicators (intermediate momentum)
    rsi_4h_14 = calculate_rsi(df_4h['close'].values, period=14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Donchian Channel for breakout detection
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume average for confirmation
    vol_avg = calculate_volume_avg(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.28
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Extract hours for session filter
    hours = np.array([extract_hour(ot) for ot in open_time])
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(rsi_4h_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H RSI MOMENTUM ===
        rsi_4h_bullish = rsi_4h_aligned[i] > 50.0
        rsi_4h_bearish = rsi_4h_aligned[i] < 50.0
        rsi_4h_strong_bull = rsi_4h_aligned[i] > 55.0
        rsi_4h_strong_bear = rsi_4h_aligned[i] < 45.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1]
        donchian_breakout_down = close[i] < donchian_lower[i-1]
        
        # === ENTRY LOGIC — REGIME + MOMENTUM + BREAKOUT ===
        new_signal = 0.0
        
        # LONG ENTRIES (simpler conditions for more trades)
        # Condition 1: Bull regime + 4h RSI bullish + Donchian breakout + volume
        if bull_regime and rsi_4h_bullish and donchian_breakout_up and volume_confirmed and in_session:
            new_signal = LONG_SIZE
        # Condition 2: Bull regime + strong HMA slope + 4h RSI strong bull
        elif bull_regime and hma_slope_bull and rsi_4h_strong_bull and in_session:
            new_signal = LONG_SIZE * 0.8
        # Condition 3: Bull regime + Donchian breakout (trend continuation)
        elif bull_regime and donchian_breakout_up and volume_confirmed:
            new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (mirror logic for bear market)
        if new_signal == 0.0:
            # Condition 1: Bear regime + 4h RSI bearish + Donchian breakdown + volume
            if bear_regime and rsi_4h_bearish and donchian_breakout_down and volume_confirmed and in_session:
                new_signal = -SHORT_SIZE
            # Condition 2: Bear regime + strong HMA slope + 4h RSI strong bear
            elif bear_regime and hma_slope_bear and rsi_4h_strong_bear and in_session:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 3: Bear regime + Donchian breakdown (trend continuation)
            elif bear_regime and donchian_breakout_down and volume_confirmed:
                new_signal = -SHORT_SIZE * 0.7
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long if regime flips bearish
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            # Exit on opposite breakout
            if close[i] < donchian_lower[i-1]:
                new_signal = 0.0
        
        # Exit short if regime flips bullish
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            # Exit on opposite breakout
            if close[i] > donchian_upper[i-1]:
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