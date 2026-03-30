#!/usr/bin/env python3
"""
Experiment #024: Dual-Mode Camarilla + Choppiness Regime (4h)

HYPOTHESIS: Markets switch between TRENDING and CHOPPING regimes.
- CHOPPING (CHOP > 61.8): Mean-revert to Camarilla S3/R3 bounds
- TRENDING (CHOP < 38.2): Breakout trades with volume confirmation
- This dual-mode approach adapts to both bull (2021) and bear (2022) markets

WHY IT SHOULD WORK:
- Camarilla pivot levels are mathematically derived support/resistance
- Choppiness Index is a proven regime detector (DB winner: test_sharpe=1.471)
- Volume spike confirms momentum validity
- Simple 2-3 conditions = achievable trade frequency
- ATR-based stoploss manages risk in volatile markets

EXPECTED TRADE COUNT: 100-200 total over 4 years (25-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_camarilla_pivots(high, low, close, open_price=None):
    """
    Camarilla Pivot Levels (classic 8 levels)
    R4 = close + (high - low) * 1.1/2
    R3 = close + (high - low) * 1.1/4
    R2 = close + (high - low) * 1.1/6
    R1 = close + (high - low) * 1.1/12
    S1 = close - (high - low) * 1.1/12
    S2 = close - (high - low) * 1.1/6
    S3 = close - (high - low) * 1.1/4
    S4 = close - (high - low) * 1.1/2
    """
    n = len(close)
    pivot_range = high - low
    
    r4 = close + pivot_range * 0.55
    r3 = close + pivot_range * 0.275
    r2 = close + pivot_range * 0.183
    r1 = close + pivot_range * 0.092
    s1 = close - pivot_range * 0.092
    s2 = close - pivot_range * 0.183
    s3 = close - pivot_range * 0.275
    s4 = close - pivot_range * 0.55
    
    return r4, r3, r2, r1, s1, s2, s3, s4

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * (log10(sum(ATR(1), period)) / log10(period * (high_rolling - low_rolling)))
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = np.zeros(n)
    atr[0] = high[0] - low[0]
    for i in range(1, n):
        atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        high_max = np.max(high[i-period+1:i+1])
        low_min = np.min(low[i-period+1:i+1])
        range_val = high_max - low_min
        
        if range_val > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * (np.log10(atr_sum) / np.log10(range_val * period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_arr = prices["open"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF indicators (1d) ===
    # HTF Camarilla pivots for structure
    r4_1d, r3_1d, r2_1d, r1_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla_pivots(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align HTF pivots to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness Index on 4h
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local EMA for short-term trend
    ema_8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    CHOP_CHOPPY = 61.8
    CHOP_TRENDING = 38.2
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 50  # Enough for choppiness and volume
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop_14[i] > CHOP_CHOPPY
        is_trending = chop_14[i] < CHOP_TRENDING
        is_neutral = not is_choppy and not is_trending
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            if is_choppy:
                # CHOPPY REGIME: Mean-reversion to Camarilla bounds
                # Long when price approaches S3 from above
                if close[i] <= s3_1d_aligned[i] * 1.005 and close[i] >= s3_1d_aligned[i] * 0.98:
                    desired_signal = SIZE
                # Short when price approaches R3 from below
                elif close[i] >= r3_1d_aligned[i] * 0.995 and close[i] <= r3_1d_aligned[i] * 1.02:
                    desired_signal = -SIZE
                    
            elif is_trending:
                # TRENDING REGIME: Momentum breakout
                # Long on bullish breakout with volume
                if close[i] > r2_1d_aligned[i] and vol_spike:
                    desired_signal = SIZE
                # Short on bearish breakdown with volume
                elif close[i] < s2_1d_aligned[i] and vol_spike:
                    desired_signal = -SIZE
                    
            else:
                # NEUTRAL REGIME: Tight ranges, skip
                pass
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 3 ATR from highest point
                stop_price = trailing_high - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price hits R3 in choppy (mean-revert target)
                if is_choppy and close[i] >= r3_1d_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 3 ATR from lowest point
                stop_price = trailing_low + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price hits S3 in choppy (mean-revert target)
                if is_choppy and close[i] <= s3_1d_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals