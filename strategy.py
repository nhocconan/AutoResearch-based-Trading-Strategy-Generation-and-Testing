#!/usr/bin/env python3
"""
Experiment #1115: 6h Primary + 12h/1d HTF — KAMA Adaptive Trend + BB Mean Reversion Hybrid

Hypothesis: 6h timeframe sits between 4h (noise) and 12h (slow). Using KAMA (Kaufman Adaptive MA)
which adjusts smoothing based on market efficiency ratio, combined with Bollinger Band mean reversion
when trend is weak, will capture both trending and ranging phases better than static HMA.

Key innovations:
1. KAMA (Kaufman Adaptive MA): ER-based smoothing adapts to volatility regimes automatically
2. BB Width Percentile: Detects squeeze/expansion for regime detection (alternative to CHOP)
3. Dual-mode entries:
   - Trend mode (BB expanding): KAMA slope + price position
   - Mean-revert mode (BB squeezed): BB band touch + RSI extreme
4. 12h KAMA as primary HTF filter (faster response than 1d for 6h primary)
5. Volume surge filter: entry volume > 1.5x 20-bar avg volume
6. Asymmetric sizing: 0.30 for strong signals, 0.20 for moderate

Why this should work on 6h:
- KAMA adapts to crypto's variable volatility better than fixed-period HMA/EMA
- BB squeeze detection catches consolidation breakouts (common on 6h)
- 12h HTF provides trend bias without being too slow (1d is too laggy for 6h entries)
- Volume filter reduces false breakouts
- Wider RSI thresholds (35/65) ensure sufficient trade frequency

Entry conditions (calibrated for 30-60 trades/year on 6h):
- LONG trend: price>12h_KAMA + KAMA sloping up + BB expanding + volume surge
- LONG mean-revert: price<BB_lower + RSI<40 + 12h_KAMA bullish + BB squeezed
- SHORT trend: price<12h_KAMA + KAMA sloping down + BB expanding + volume surge
- SHORT mean-revert: price>BB_upper + RSI>60 + 12h_KAMA bearish + BB squeezed

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_bb_hybrid_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]):
            net_change = abs(close[i] - close[i - er_period])
            sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if sum_changes > 1e-10:
                er[i] = net_change / sum_changes
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    # Initialize at first valid ER point
    init_idx = er_period
    kama[init_idx] = close[init_idx]
    
    for i in range(init_idx + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with width calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Rolling mean and std
    ma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    width = upper - lower
    
    return upper, lower, ma, width

def calculate_bb_width_percentile(width, lookback=100):
    """Percentile rank of BB width over lookback period"""
    n = len(width)
    pct = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window = width[i - lookback + 1:i + 1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < width[i])
            pct[i] = 100.0 * count_below / (lookback - 1)
    
    return pct

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_surge(volume, period=20, threshold=1.5):
    """Detect volume surge above threshold * average"""
    n = len(volume)
    if n < period:
        return np.zeros(n, dtype=bool)
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    surge = volume > (threshold * avg_vol)
    surge[:period] = False
    return surge

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA
    kama_12h_raw = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bb_ma, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_surge = calculate_volume_surge(volume, period=20, threshold=1.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (BB Width Percentile) ===
        is_squeeze = bb_width_pct[i] < 30.0  # BB at low width = consolidation
        is_expansion = bb_width_pct[i] > 70.0  # BB expanding = trending
        
        # === HTF TREND BIAS ===
        htf_bullish = close[i] > kama_12h_aligned[i]
        htf_bearish = close[i] < kama_12h_aligned[i]
        
        # KAMA slope (6h)
        kama_slope = 0.0
        if i >= 3 and not np.isnan(kama_6h[i - 3]):
            kama_slope = kama_6h[i] - kama_6h[i - 3]
        
        kama_sloping_up = kama_slope > 0
        kama_sloping_down = kama_slope < 0
        
        # === ENTRY LOGIC (DUAL MODE) ===
        desired_signal = 0.0
        
        # TREND MODE (BB expanding)
        if is_expansion:
            # Long: HTF bullish + KAMA sloping up + price above KAMA + volume surge
            if htf_bullish and kama_sloping_up and close[i] > kama_6h[i] and vol_surge[i]:
                desired_signal = SIZE_STRONG
            elif htf_bullish and kama_sloping_up and close[i] > kama_6h[i]:
                desired_signal = SIZE_BASE
            
            # Short: HTF bearish + KAMA sloping down + price below KAMA + volume surge
            elif htf_bearish and kama_sloping_down and close[i] < kama_6h[i] and vol_surge[i]:
                desired_signal = -SIZE_STRONG
            elif htf_bearish and kama_sloping_down and close[i] < kama_6h[i]:
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION MODE (BB squeezed)
        elif is_squeeze:
            # Long: price at/near lower BB + RSI oversold + HTF not strongly bearish
            if close[i] <= bb_lower[i] * 1.005 and rsi_14[i] < 40 and not htf_bearish:
                desired_signal = SIZE_BASE
            # Stronger long: deeper BB touch + very oversold RSI
            elif close[i] <= bb_lower[i] * 1.01 and rsi_14[i] < 30 and htf_bullish:
                desired_signal = SIZE_STRONG
            
            # Short: price at/near upper BB + RSI overbought + HTF not strongly bullish
            elif close[i] >= bb_upper[i] * 0.995 and rsi_14[i] > 60 and not htf_bullish:
                desired_signal = -SIZE_BASE
            # Stronger short: deeper BB touch + very overbought RSI
            elif close[i] >= bb_upper[i] * 0.99 and rsi_14[i] > 70 and htf_bearish:
                desired_signal = -SIZE_STRONG
        
        # NEUTRAL MODE (BB normal) - use KAMA cross
        else:
            # Long: price crosses above KAMA + HTF bullish
            if close[i] > kama_6h[i] and htf_bullish and rsi_14[i] > 45:
                desired_signal = SIZE_BASE
            # Short: price crosses below KAMA + HTF bearish
            elif close[i] < kama_6h[i] and htf_bearish and rsi_14[i] < 55:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals