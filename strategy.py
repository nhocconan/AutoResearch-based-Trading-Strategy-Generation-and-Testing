#!/usr/bin/env python3
"""
Experiment #074: 4h Primary + 12h/1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Replace CRSI/Choppiness regime with simpler ADX + KAMA trend filter.
Fisher Transform catches reversals better than RSI in bear markets (proven in #071).
KAMA adapts to volatility better than HMA/EMA. Simpler logic = more trades.

Key changes from #064:
1) Fisher Transform (period=9) instead of CRSI — better reversal detection
2) KAMA (ER=10) instead of HMA — adaptive to market efficiency
3) ADX(14) > 20 for trend confirmation instead of Choppiness
4) Remove volume spike filter — was too restrictive (caused 0 trades)
5) Simpler hold logic — hold while Fisher confirms direction
6) Asymmetric entries: with HTF trend = easier entry, against = harder

Why this should work:
- 4h proven timeframe
- Fisher Transform worked in #071 (Sharpe=0.410, Return=+92.6%)
- KAMA reduces whipsaw in ranging markets
- Simpler filters = more trades (avoid 0-trade failure)
- HTF bias prevents counter-trend blowups in 2022 crash

Position size: 0.25-0.30 (discrete)
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.486
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_adx_regime_12h1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market efficiency ratio (ER).
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close_s.diff(er_period).values)
    vol_sum = np.abs(close_s.diff().values)
    
    er = np.zeros(n)
    for i in range(er_period, n):
        vol_window = vol_sum[i-er_period+1:i+1]
        if vol_window.sum() > 0:
            er[i] = price_change[i] / vol_window.sum()
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = close_s.iloc[er_period]
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for better reversal detection.
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price position
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        x = (2.0 * (high[i] + low[i]) / 2.0 - (hh + ll)) / range_val
        
        # Clamp to avoid division issues
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger line (1-period lag of fisher)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth +DM, -DM, TR using Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100.0 * minus_dm_s[i] / tr_s[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA for intermediate trend
    kama_12h = calculate_kama(df_12h['close'].values, er_period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1d KAMA for macro bias
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_4h = calculate_kama(close, er_period=10)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_TREND = 0.30
    POSITION_SIZE_COUNTER = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === HTF TREND BIAS ===
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # Strong bullish: both 12h and 1d above KAMA
        htf_bullish = price_above_kama_12h and price_above_kama_1d
        # Strong bearish: both 12h and 1d below KAMA
        htf_bearish = price_below_kama_12h and price_below_kama_1d
        
        # === TREND STRENGTH ===
        trend_strength = adx[i]
        is_trending = trend_strength > 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above trigger from below -2.0
        fisher_long = fisher[i] > fisher_trigger[i] and fisher_trigger[i] < -1.5
        # Short: Fisher crosses below trigger from above +2.0
        fisher_short = fisher[i] < fisher_trigger[i] and fisher_trigger[i] > 1.5
        
        # === KAMA DIRECTION ===
        kama_rising = kama_4h[i] > kama_4h[i-1] if i > 0 else False
        kama_falling = kama_4h[i] < kama_4h[i-1] if i > 0 else False
        
        # === ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- WITH HTF TREND (easier entry) ---
        if htf_bullish:
            # Long with trend: Fisher long + KAMA rising + ADX confirms
            if fisher_long and kama_rising:
                if is_trending or adx[i] > 15.0:
                    new_signal = POSITION_SIZE_TREND
        
        elif htf_bearish:
            # Short with trend: Fisher short + KAMA falling + ADX confirms
            if fisher_short and kama_falling:
                if is_trending or adx[i] > 15.0:
                    new_signal = -POSITION_SIZE_TREND
        
        # --- AGAINST HTF TREND (harder entry - mean reversion) ---
        else:
            # Ranging market: require stronger Fisher signal
            if fisher_long and fisher[i] < -2.0:
                new_signal = POSITION_SIZE_COUNTER
            elif fisher_short and fisher[i] > 2.0:
                new_signal = -POSITION_SIZE_COUNTER
        
        # === HOLD POSITION LOGIC ===
        # Hold long if Fisher still positive or rising
        if in_position and position_side > 0 and new_signal == 0.0:
            if fisher[i] > -1.0 or (fisher[i] > fisher_trigger[i]):
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # Hold short if Fisher still negative or falling
        if in_position and position_side < 0 and new_signal == 0.0:
            if fisher[i] < 1.0 or (fisher[i] < fisher_trigger[i]):
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
        
        # === EXIT ON HTF TREND CHANGE ===
        if in_position and position_side > 0:
            if htf_bearish:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if htf_bullish:
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