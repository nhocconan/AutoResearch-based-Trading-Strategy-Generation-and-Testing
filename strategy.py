#!/usr/bin/env python3
"""
Experiment #010: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime

Hypothesis: 1h timeframe with 4h trend bias + Fisher reversals + session filter
will generate 40-80 trades/year with positive Sharpe.

Key components:
1. Ehlers Fisher Transform (period=9): Catches reversals in bear/range markets
2. Choppiness Index (14): Regime detection - CHOP>55 range, CHOP<45 trend
3. 4h HMA(21): Macro trend bias (only trade with 4h trend)
4. Session filter: Only 8-20 UTC (high volume hours, reduces trades)
5. Volume filter: volume > 0.8x 20-period average
6. ATR(14) stoploss: 2.5*ATR trailing

Why this should work:
- 1h primary = enough signals for entries but not too many
- 4h HTF = strong trend filter, avoids counter-trend trades
- Fisher Transform = better reversal detection than RSI in choppy markets
- Session filter = naturally reduces trade count to 40-80/year
- LOOSE Fisher thresholds (-1.5/+1.5) = ensures trade generation

Position size: 0.25 (smaller for 1h TF, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_session_4h_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversals.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalize price to -1 to +1 range
        normalized = 0.667 * ((typical[i] - lowest) / range_val - 0.5) + 0.67 * fisher_signal[i-1]
        
        # Clamp to prevent division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized)) + 0.5 * fisher[i-1]
        fisher_signal[i] = normalized
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def extract_hour_from_timestamp(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    timestamps = prices["open_time"].values / 1000.0
    hours = (timestamps % 86400) / 3600.0
    return hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    hours = extract_hour_from_timestamp(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for macro bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h HMA slope (trend direction)
    hma_4h_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-5]):
            hma_4h_slope[i] = hma_4h_aligned[i] - hma_4h_aligned[i-5]
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(ema_21[i]):
            continue
        if np.isnan(vol_avg_20[i]) or atr_14[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 4H MACRO BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        hma_4h_bullish = hma_4h_slope[i] > 0
        hma_4h_bearish = hma_4h_slope[i] < 0
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market
        is_trending = chop_value < 45.0  # Trend market
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds for trade gen) ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # Fisher cross above -1.5 (bullish reversal)
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        # Fisher cross below +1.5 (bearish reversal)
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # === EMA TREND ===
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours with volume
        if in_session and volume_ok:
            # --- RANGING REGIME: Mean Reversion with Fisher ---
            if is_ranging:
                # Long: Fisher oversold + 4h not strongly bearish
                if fisher_oversold or fisher_cross_up:
                    if not hma_4h_bearish or price_above_hma_4h:
                        new_signal = POSITION_SIZE
                
                # Short: Fisher overbought + 4h not strongly bullish
                elif fisher_overbought or fisher_cross_down:
                    if not hma_4h_bullish or price_below_hma_4h:
                        new_signal = -POSITION_SIZE
            
            # --- TRENDING REGIME: Trend Following with Fisher pullback ---
            elif is_trending:
                # Long: 4h bullish + Fisher pulling back from oversold
                if hma_4h_bullish and price_above_hma_4h:
                    if fisher_rising and fisher[i] > -1.0:
                        new_signal = POSITION_SIZE
                
                # Short: 4h bearish + Fisher pulling back from overbought
                elif hma_4h_bearish and price_below_hma_4h:
                    if fisher_falling and fisher[i] < 1.0:
                        new_signal = -POSITION_SIZE
            
            # --- FALLBACK: EMA + Fisher confluence ---
            if new_signal == 0.0:
                # Long: EMA bullish + Fisher crossing up
                if ema_bullish and fisher_cross_up:
                    new_signal = POSITION_SIZE
                
                # Short: EMA bearish + Fisher crossing down
                elif ema_bearish and fisher_cross_down:
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
        
        # === EXIT ON REGIME/TREND CHANGE ===
        # Exit long if 4h trend turns strongly bearish
        if in_position and position_side > 0:
            if hma_4h_bearish and price_below_hma_4h and chop_value < 45:
                new_signal = 0.0
        
        # Exit short if 4h trend turns strongly bullish
        if in_position and position_side < 0:
            if hma_4h_bullish and price_above_hma_4h and chop_value < 45:
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