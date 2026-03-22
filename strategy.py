#!/usr/bin/env python3
"""
Experiment #607: 1d Primary + 1w HTF — KAMA Adaptive Trend + Choppiness Regime + Asymmetric RSI

Hypothesis: Building on #604 success (4h KAMA+CHOP+RSI, Sharpe=0.378) and current best 
mtf_1d_chop_crsi_regime_1w_v1 (Sharpe=0.520), this strategy combines adaptive KAMA trend 
following with regime-switching logic and ASYMMETRIC entry rules for crypto markets.

Key insights from 537 failed strategies:
1. Crypto is asymmetric: sharp drops (-77% in 2022), slow rallies. Symmetric rules fail.
2. KAMA adapts to volatility better than EMA/HMA (proven in #604)
3. 1w HTF trend filter prevents counter-trend trades during major moves
4. Choppiness Index correctly identifies trend vs range regimes
5. Asymmetric RSI: longs need deeper oversold (25-45), shorts need moderate overbought (55-75)
   because crashes are faster than rallies

Why this might beat Sharpe=0.520:
- KAMA on both 1d and 1w reduces whipsaw vs static MAs
- 1w trend filter (KAMA slope) keeps us on right side of major moves
- Regime-switching: trend-follow when CHOP<45, mean-revert when CHOP>55
- Asymmetric RSI entries match crypto market behavior
- Conservative size (0.28) controls drawdown through 2022 crash
- 2.5*ATR trailing stop limits losses on fast reversals

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 20-40 trades/year on 1d (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_rsi_asym_1w_v1"
timeframe = "1d"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over ER period
    price_change = np.abs(close_s.diff(er_period).values)
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Efficiency Ratio (ER)
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:er_period] = np.nan
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA for primary trend direction
    kama_1w = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (KAMA slope over 3 bars) ===
        kama_1w_slope_bull = kama_1w_aligned[i] > kama_1w_aligned[i-3] if i >= 3 else False
        kama_1w_slope_bear = kama_1w_aligned[i] < kama_1w_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1w KAMA
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === 1D KAMA SLOPE (2 bars) ===
        kama_1d_slope_bull = kama_1d[i] > kama_1d[i-2] if i >= 2 else False
        kama_1d_slope_bear = kama_1d[i] < kama_1d[i-2] if i >= 2 else False
        
        # Price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d[i]
        price_below_kama_1d = close[i] < kama_1d[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0
        is_chop_regime = chop_14[i] > 55.0
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 1w trend with 1d pullback entries ---
        if is_trend_regime:
            # LONG: 1w bull + 1d bull + price above both KAMAs + RSI pullback (asymmetric: 30-50)
            if kama_1w_slope_bull and kama_1d_slope_bull and price_above_kama_1w and price_above_kama_1d:
                if 30.0 <= rsi_14[i] <= 50.0:
                    new_signal = POSITION_SIZE
            
            # SHORT: 1w bear + 1d bear + price below both KAMAs + RSI bounce (asymmetric: 50-70)
            elif kama_1w_slope_bear and kama_1d_slope_bear and price_below_kama_1w and price_below_kama_1d:
                if 50.0 <= rsi_14[i] <= 70.0:
                    new_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at RSI extremes (asymmetric) ---
        elif is_chop_regime:
            # LONG: RSI < 35 (deep oversold for crypto) + price below 1d KAMA
            if rsi_14[i] < 35.0 and price_below_kama_1d:
                new_signal = POSITION_SIZE
            
            # SHORT: RSI > 65 (moderate overbought) + price above 1d KAMA
            elif rsi_14[i] > 65.0 and price_above_kama_1d:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if kama_1w_slope_bear and price_below_kama_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_1w_slope_bull and price_above_kama_1w:
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