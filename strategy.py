#!/usr/bin/env python3
"""
Experiment #045: 1h Primary + 4h/1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Based on research showing Ehlers Fisher Transform excels at catching reversals
in bear/range markets (like 2025), combined with KAMA's adaptive trend following that
reduces whipsaw during volatile periods. This differs from CRSI-based approaches.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, catches reversals at extremes (-1.5/+1.5 levels)
2. KAMA (Kaufman Adaptive): ER-based smoothing that adapts to market efficiency
3. CHOPPINESS INDEX: regime filter (CHOP>55=range, CHOP<45=trend)
4. 4h HMA: higher timeframe trend bias (only trade with 4h trend)
5. SESSION FILTER: only 8-20 UTC (high liquidity periods)
6. VOLUME CONFIRMATION: volume > 0.8x 20-period avg

Why 1h works with strict filters:
- 4h HMA provides trend direction (reduces false signals)
- Fisher Transform gives precise entry timing
- Session filter eliminates low-liquidity whipsaws
- Target: 40-70 trades/year (fee-efficient)

Entry conditions (balanced for trade generation):
- Long: Fisher < -1.2 + KAMA bullish + 4h HMA bullish + CHOP regime OK + session + volume
- Short: Fisher > +1.2 + KAMA bearish + 4h HMA bearish + CHOP regime OK + session + volume

Position size: 0.25 (conservative for 1h timeframe)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_chop_session_4h1d_v1"
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
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close_s.diff().values)
    price_change[0] = 0
    
    sum_price_change = pd.Series(price_change).rolling(window=er_period, min_periods=er_period).sum().values
    
    net_change = np.abs(close_s.diff(er_period).values)
    net_change[:er_period] = 0
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = sum_price_change > 0
    er[mask] = net_change[mask] / sum_price_change[mask]
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    sc[mask] = (er[mask] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer signals.
    Entry when Fisher crosses above -1.5 (long) or below +1.5 (short).
    """
    n = len(close)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        # Normalize price to -1 to +1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        normalized = 2.0 * (typical[i] - lowest) / range_val - 1.0
        
        # Clamp to avoid extreme values
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers method)
        if i > period:
            fisher[i] = 0.7 * fisher[i] + 0.3 * fisher_prev[i-1]
        
        fisher_prev[i] = fisher[i]
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_1h[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(volume_ma[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * volume_ma[i]
        
        # === 4H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope (3-bar lookback for trend confirmation)
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_slope_bull = kama_1h[i] > kama_1h[i-3] if i >= 3 else False
        kama_slope_bear = kama_1h[i] < kama_1h[i-3] if i >= 3 else False
        price_above_kama = close[i] > kama_1h[i]
        price_below_kama = close[i] < kama_1h[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.2  # Long entry zone
        fisher_overbought = fisher[i] > 1.2  # Short entry zone
        
        # Fisher cross (for timing precision)
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi_not_extreme_long = rsi_14[i] > 25  # Not capitulation
        rsi_not_extreme_short = rsi_14[i] < 75  # Not euphoria
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY (3+ confluence required) ---
        long_conditions = 0
        
        # Must have 4h trend support OR macro support
        if price_above_hma_4h or price_above_hma_1d:
            long_conditions += 1
        
        # Fisher oversold or cross
        if fisher_oversold or fisher_cross_up:
            long_conditions += 1
        
        # KAMA bullish
        if kama_slope_bull and price_above_kama:
            long_conditions += 1
        
        # Session + volume
        if in_session and volume_ok:
            long_conditions += 1
        
        # RSI filter
        if rsi_not_extreme_long:
            long_conditions += 1
        
        # Need 4+ conditions for long entry
        if long_conditions >= 4:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY (3+ confluence required) ---
        short_conditions = 0
        
        # Must have 4h trend support OR macro support
        if price_below_hma_4h or price_below_hma_1d:
            short_conditions += 1
        
        # Fisher overbought or cross
        if fisher_overbought or fisher_cross_down:
            short_conditions += 1
        
        # KAMA bearish
        if kama_slope_bear and price_below_kama:
            short_conditions += 1
        
        # Session + volume
        if in_session and volume_ok:
            short_conditions += 1
        
        # RSI filter
        if rsi_not_extreme_short:
            short_conditions += 1
        
        # Need 4+ conditions for short entry
        if short_conditions >= 4 and new_signal == 0.0:
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
            if price_below_hma_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short if 4h trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_bull:
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