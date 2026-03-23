#!/usr/bin/env python3
"""
Experiment #058: 30m Primary + 4h/1d HTF — Fisher Transform + Dual HMA Trend + Volume

Hypothesis: 30m entries with 4h+1d dual trend filter using Ehlers Fisher Transform
for reversal timing will generate 40-80 trades/year with Sharpe > 0.486.

Key insights from 50+ failed experiments:
1) 30m strategies fail due to TOO MANY trades (>200/yr) → fee drag kills profit
2) Single HTF (4h only) not strict enough for lower TF
3) Need DUAL HTF filter (4h + 1d both agree) for signal direction
4) Fisher Transform proven in bear/range markets (not tried in failed 30m strats)
5) Session filter (8-20 UTC) reduces whipsaws during low liquidity
6) Volume spike confirmation (>1.2x avg) filters false breakouts

Why this should work:
- 30m primary = faster entries than 4h/1d but still manageable trade count
- 4h + 1d DUAL HMA = very strict trend filter (both must agree)
- Fisher Transform = catches reversals at extremes (proven in research notes)
- Choppiness regime = adapts between trend-follow and mean-revert
- Volume + Session = filters low-quality signals during thin markets
- Position size 0.25 = smaller for lower TF (controls drawdown)

Entry confluence (ALL must agree for long):
1) 4h HMA bullish (price > hma_4h)
2) 1d HMA bullish (price > hma_1d_aligned)
3) Fisher < -1.0 (oversold extreme)
4) Choppiness < 50 (trending regime, not choppy)
5) Volume > 1.2x 20-bar average
6) Hour 8-20 UTC (liquid session)

Position size: 0.25 (discrete, smaller for 30m TF)
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_dualhma_chop_vol_session_v1"
timeframe = "30m"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - catches reversals at extremes.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    # Calculate median price
    median = (high + low) / 2.0
    
    # Normalize price to range -1 to +1
    highest = pd.Series(median).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(median).rolling(window=period, min_periods=period).min().values
    price_range = highest - lowest + 1e-10
    
    normalized = (2.0 * (median - lowest) / price_range) - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)  # Prevent log domain errors
    
    # Apply Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
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
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time_array // (1000 * 60 * 60)) % 24
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
    
    # Calculate 4h HMA for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias (dual HTF filter)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Smaller for 30m TF (controls fee drag)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_ratio[i]) or atr_14[i] == 0:
            continue
        
        # === 4H TREND (primary direction) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D MACRO BIAS (dual HTF confirmation) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === DUAL HTF CONFLUENCE (both must agree) ===
        dual_bullish = price_above_hma_4h and price_above_hma_1d
        dual_bearish = price_below_hma_4h and price_below_hma_1d
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_oversold = fisher[i] < -1.0  # Extreme oversold
        fisher_overbought = fisher[i] > 1.0  # Extreme overbought
        
        # Fisher crossover signals (more reliable than absolute levels)
        fisher_cross_up = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_trending = chop_value < 50.0  # Trending market (not choppy)
        is_ranging = chop_value > 61.8  # Range market
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        # === SESSION FILTER (8-20 UTC - liquid hours) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === ENTRY SIGNALS (ALL confluence must agree) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Dual bullish + Fisher reversal + Volume + Session ---
        if dual_bullish and is_trending:
            if (fisher_oversold or fisher_cross_up) and volume_confirmed and in_session:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Dual bearish + Fisher reversal + Volume + Session ---
        elif dual_bearish and is_trending:
            if (fisher_overbought or fisher_cross_down) and volume_confirmed and in_session:
                new_signal = -POSITION_SIZE
        
        # --- RANGE REGIME: Mean reversion at extremes (stricter) ---
        elif is_ranging:
            # Long in range: very oversold + dual bullish bias
            if fisher[i] < -2.0 and price_above_hma_4h and volume_confirmed:
                new_signal = POSITION_SIZE * 0.5  # Half size in range
            
            # Short in range: very overbought + dual bearish bias
            elif fisher[i] > 2.0 and price_below_hma_4h and volume_confirmed:
                new_signal = -POSITION_SIZE * 0.5  # Half size in range
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if Fisher not at opposite extreme
            if position_side > 0 and fisher[i] < 1.5:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and fisher[i] > -1.5:
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
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            # Exit if both HTF turn bearish
            if price_below_hma_4h and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit if both HTF turn bullish
            if price_above_hma_4h and price_above_hma_1d:
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