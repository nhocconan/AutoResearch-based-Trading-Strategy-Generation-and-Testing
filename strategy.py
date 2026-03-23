#!/usr/bin/env python3
"""
Experiment #050: 1h Primary + 4h/12h HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Based on research showing Ehlers Fisher Transform excels at catching reversals 
in bear/range markets (unlike RSI which lags), combined with KAMA's adaptive smoothing 
that reduces whipsaw during low-volatility periods. This is DIFFERENT from the 46+ 
failed CRSI/CHOP strategies.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, triggers at -1.5/+1.5 levels (proven reversal catcher)
2. KAMA (Kaufman Adaptive): ER-based smoothing that adapts to market efficiency
3. 4h HMA for trend direction (HTF bias)
4. 12h HMA for macro filter (only trade with macro trend)
5. Session filter: 8-20 UTC only (avoid Asia low-liquidity whipsaw)
6. Volume confirmation: >0.8x 20-period average

Why 1h works with HTF:
- 4h/12h determine DIRECTION (reduces false signals)
- 1h Fisher determines ENTRY TIMING (precision within HTF trend)
- Target: 40-70 trades/year (fee-efficient per Rule 10)
- Session filter cuts ~40% of low-quality signals

Entry conditions (balanced for trade generation):
- Long: Fisher < -1.2 + Fisher turning up + 4h HMA bullish + 12h HMA bullish + session + volume
- Short: Fisher > +1.2 + Fisher turning down + 4h HMA bearish + 12h HMA bearish + session + volume

Position size: 0.25 (conservative for 1h timeframe)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_session_4h12h_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close_s.diff(er_period).values)
    sum_volatility = np.abs(close_s.diff().values)
    
    # Rolling sum of absolute differences
    volatility_sum = pd.Series(sum_volatility).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = volatility_sum > 0
    er[mask] = price_change / volatility_sum[mask]
    er[:er_period] = np.nan
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    sc[mask] = (er[mask] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price
    Excellent for catching reversals in ranging/bear markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Normalize price using highest high and lowest low over period
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            trigger[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1 range
        x = (2.0 * (close[i] - lowest) / price_range) - 1.0
        
        # Clamp to avoid log domain errors
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger is previous Fisher value
        trigger[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, trigger

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
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for macro bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume moving average for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_1h[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        if atr_14[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === 12H MACRO BIAS ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H TREND BIAS ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-5] if i >= 5 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-5] if i >= 5 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA TREND CONFIRMATION ===
        kama_slope_bull = kama_1h[i] > kama_1h[i-5] if i >= 5 else False
        kama_slope_bear = kama_1h[i] < kama_1h[i-5] if i >= 5 else False
        price_above_kama = close[i] > kama_1h[i]
        price_below_kama = close[i] < kama_1h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        fisher_turning_up = fisher[i] > fisher_trigger[i]
        fisher_turning_down = fisher[i] < fisher_trigger[i]
        
        # === RSI FILTER (avoid extreme overbought/oversold against trend) ===
        rsi_not_extreme_long = rsi_14[i] < 75  # Don't long at extreme overbought
        rsi_not_extreme_short = rsi_14[i] > 25  # Don't short at extreme oversold
        
        # === ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Fisher oversold + turning up + HTF bullish confluence ---
        if fisher_oversold and fisher_turning_up:
            # Require 4h trend bullish + 12h macro bullish (strong confluence)
            if hma_4h_slope_bull and price_above_hma_4h:
                if price_above_hma_12h and kama_slope_bull:
                    if in_session and volume_ok and rsi_not_extreme_long:
                        new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Fisher overbought + turning down + HTF bearish confluence ---
        elif fisher_overbought and fisher_turning_down:
            # Require 4h trend bearish + 12h macro bearish (strong confluence)
            if hma_4h_slope_bear and price_below_hma_4h:
                if price_below_hma_12h and kama_slope_bear:
                    if in_session and volume_ok and rsi_not_extreme_short:
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