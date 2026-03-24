#!/usr/bin/env python3
"""
Experiment #215: 6h Primary + 12h/1d HTF — Vol Spike Reversion + Asymmetric Regime

Hypothesis: 6h timeframe captures multi-day volatility cycles better than 4h or 12h.
Vol spike reversion (ATR(7)/ATR(30) > 2.0 + price < BB lower) has proven edge in 
panic reversals. Combined with asymmetric regime (only short in bear markets via 
ADX + SMA50 filter), this should work through 2022 crash AND 2025 bear market.

Key innovations vs failed 6h attempts:
1. Vol spike ratio (ATR short/long) instead of absolute ATR
2. Asymmetric short logic (ADX>25 + price<SMA50 = bear regime, only short then)
3. ADX hysteresis (enter 25, exit 18) to reduce whipsaw
4. 12h HMA for intermediate trend, 1d HMA for major bias
5. Bollinger Band %B for precise mean reversion entry

Target: 25-45 trades/year, Sharpe>0.40 (beat current 6h best of 0.399)
Position size: 0.25 base, 0.30 for vol spike entries
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_asymmetric_regime_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100.0 * minus_dm_s[i] / tr_s[i]
    
    # DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with %B indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # %B = (price - lower) / (upper - lower)
    pct_b = np.zeros(n)
    pct_b[:] = np.nan
    for i in range(period, n):
        band_width = upper[i] - lower[i]
        if band_width > 1e-10:
            pct_b[i] = (close[i] - lower[i]) / band_width
    
    return upper, lower, pct_b

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    adx = calculate_adx(high, low, close, period=14)
    
    bb_upper, bb_lower, pct_b = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    adx_entry_state = 0  # Track ADX hysteresis state
    
    for i in range(300, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_14[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_lower[i]) or np.isnan(pct_b[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY SPIKE RATIO ===
        vol_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 1e-10 else 0.0
        vol_spike = vol_ratio > 2.0  # Extreme volatility
        vol_normalizing = vol_ratio < 1.2  # Volatility returning to normal
        
        # === REGIME DETECTION ===
        # Bear regime: ADX > 25 + price < SMA50 (trending down)
        # Use hysteresis: enter bear at ADX>25, exit at ADX<18
        in_bear_regime = adx[i] > 25.0 and close[i] < sma_50[i]
        exiting_bear_regime = adx[i] < 18.0
        
        # Bull regime: price > SMA200
        in_bull_regime = close[i] > sma_200[i]
        
        # === HTF TREND BIAS ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === BOLLINGER %B EXTREMES ===
        bb_extreme_low = pct_b[i] < 0.05  # Price at or below lower band
        bb_extreme_high = pct_b[i] > 0.95  # Price at or above upper band
        bb_mean_revert_long = pct_b[i] < 0.2  # Recovering from extreme
        bb_mean_revert_short = pct_b[i] > 0.8  # Recovering from extreme
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Vol spike mean reversion (panic selling reversal)
        # Require: vol spike + BB extreme low + HTF not strongly bearish
        if vol_spike and bb_extreme_low:
            if htf_1d_bull or (htf_12h_bull and close[i] > sma_50[i]):
                desired_signal = SIZE_STRONG
            elif close[i] > sma_200[i]:  # Above long-term MA even if HTF weak
                desired_signal = SIZE_BASE
        
        # LONG: BB mean reversion in bull regime
        elif in_bull_regime and bb_mean_revert_long and not vol_spike:
            if htf_12h_bull:
                desired_signal = SIZE_BASE
        
        # SHORT: Only in bear regime (asymmetric - never short in bull)
        # Require: ADX > 25 + price < SMA50 + BB extreme high or vol spike high
        if in_bear_regime:
            if bb_extreme_high:
                if htf_1d_bear or htf_12h_bear:
                    desired_signal = -SIZE_STRONG
            elif vol_spike and pct_b[i] > 0.8:
                if htf_12h_bear:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        # Exit long when volatility normalizes (vol spike trade complete)
        if in_position and position_side > 0 and vol_normalizing:
            exit_signal = True
        
        # Exit short when bear regime ends (ADX hysteresis)
        if in_position and position_side < 0 and exiting_bear_regime:
            exit_signal = True
        
        # Exit any position when BB reaches opposite extreme (mean reversion complete)
        if in_position and position_side > 0 and pct_b[i] > 0.85:
            exit_signal = True
        if in_position and position_side < 0 and pct_b[i] < 0.15:
            exit_signal = True
        
        if stoploss_triggered or exit_signal:
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
        
        signals[i] = final_signal
    
    return signals