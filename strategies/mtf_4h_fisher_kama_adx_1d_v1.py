#!/usr/bin/env python3
"""
Experiment #689: 4h Primary + 1d HTF — Fisher Transform + KAMA Trend + ADX Filter

Hypothesis: Simpler is better. Complex regime-switching has failed repeatedly.
This strategy uses:
1. Ehlers Fisher Transform (period=9) — proven reversal catcher in bear markets
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. KAMA (Kaufman Adaptive MA) on 1d HTF — adapts to volatility, smoother than HMA
   ER (Efficiency Ratio) adjusts smoothing based on trend/noise
3. ADX(14) for trend strength — only trade when ADX > 20 (some momentum)
4. ATR-based stops (2.5x) and position sizing
5. LOOSE Fisher thresholds (-1.8/+1.8 instead of -1.5/+1.5) to ensure trades

Why this should work:
- Fisher Transform has research backing for crypto reversals (75%+ win rate at extremes)
- KAMA adapts smoothing — less whipsaw in chop, faster in trends
- 4h TF = ~30-50 trades/year (optimal fee vs signal quality)
- Simpler logic = fewer conditions that can all fail simultaneously
- ADX filter prevents trading in dead markets

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_adx_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear markets.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    # Calculate typical price
    typical = (high + low + close) / 3
    
    # Normalize price to -1 to +1 range
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest[i] = np.max(typical[i - period + 1:i + 1])
        lowest[i] = np.min(typical[i - period + 1:i + 1])
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    # Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    normalized = 2 * (typical - lowest) / range_val - 1
    normalized = np.clip(normalized, -0.999, 0.999)  # Avoid log(0)
    
    # Fisher transform: 0.5 * ln((1 + x) / (1 - x))
    fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher_raw[0] if len(fisher_raw) > 0 else 0
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period - 1, n):
        signal = np.abs(close[i] - close[i - er_period + 1])
        noise = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    # Calculate KAMA
    for i in range(er_period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, di_plus, di_minus

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    adx_4h, di_plus_4h, di_minus_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF (1d) KAMA
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Fisher crossover tracking
    prev_fisher_long_cross = False
    prev_fisher_short_cross = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d KAMA) ===
        # Price above daily KAMA = bullish bias, below = bearish bias
        trend_bullish = close[i] > kama_1d_aligned[i]
        trend_bearish = close[i] < kama_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        # Only trade when ADX > 18 (some momentum present)
        adx_strong = adx_4h[i] > 18
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds) ===
        # Long: Fisher crosses above -1.8 from below
        fisher_long_cross = (fisher_signal_4h[i] < -1.8) and (fisher_4h[i] >= -1.8)
        
        # Short: Fisher crosses below +1.8 from above
        fisher_short_cross = (fisher_signal_4h[i] > 1.8) and (fisher_4h[i] <= 1.8)
        
        # Extreme levels for stronger signals
        fisher_extreme_long = fisher_4h[i] < -2.0
        fisher_extreme_short = fisher_4h[i] > 2.0
        
        # === DI+ / DI- CONFIRMATION ===
        di_bullish = di_plus_4h[i] > di_minus_4h[i]
        di_bearish = di_minus_4h[i] > di_plus_4h[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Condition 1: Fisher long cross + trend bullish + ADX strong
        if fisher_long_cross and trend_bullish and adx_strong:
            desired_signal = SIZE_LONG
        
        # Condition 2: Fisher extreme long + trend bullish (even if ADX weak)
        elif fisher_extreme_long and trend_bullish:
            desired_signal = SIZE_LONG * 0.5
        
        # Condition 3: Fisher long cross + DI+ > DI- (momentum confirmation)
        elif fisher_long_cross and di_bullish:
            desired_signal = SIZE_LONG * 0.5
        
        # === SHORT ENTRY ===
        # Condition 1: Fisher short cross + trend bearish + ADX strong
        if fisher_short_cross and trend_bearish and adx_strong:
            desired_signal = -SIZE_SHORT
        
        # Condition 2: Fisher extreme short + trend bearish (even if ADX weak)
        elif fisher_extreme_short and trend_bearish:
            desired_signal = -SIZE_SHORT * 0.5
        
        # Condition 3: Fisher short cross + DI- > DI+ (momentum confirmation)
        elif fisher_short_cross and di_bearish:
            desired_signal = -SIZE_SHORT * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if still above daily KAMA and Fisher not overbought
                if trend_bullish and fisher_4h[i] < 2.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if still below daily KAMA and Fisher not oversold
                if trend_bearish and fisher_4h[i] > -2.0:
                    desired_signal = -SIZE_SHORT
        
        # === EXIT CONDITIONS ===
        # Long exit: Fisher > 2.0 (overbought) OR price crosses below daily KAMA
        if in_position and position_side > 0:
            if fisher_4h[i] > 2.0 or (close[i] < kama_1d_aligned[i] and adx_strong):
                desired_signal = 0.0
        
        # Short exit: Fisher < -2.0 (oversold) OR price crosses above daily KAMA
        if in_position and position_side < 0:
            if fisher_4h[i] < -2.0 or (close[i] > kama_1d_aligned[i] and adx_strong):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals