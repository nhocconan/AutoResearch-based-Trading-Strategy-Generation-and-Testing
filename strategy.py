#!/usr/bin/env python3
"""
Experiment #639: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX Filter + Asymmetric RSI

Hypothesis: Recent 4h failures (#629, #634) had too many confluence filters causing
low trade frequency. This strategy simplifies entry logic while keeping HTF trend filter.

Key changes from failed #624:
1. KAMA instead of HMA — adapts to volatility, fewer whipsaws in chop
2. ADX(14) > 20 filter — ensures trend exists but not too strict (>25 kills trades)
3. Asymmetric RSI — long when RSI<50 in uptrend, short when RSI>50 in downtrend
4. Simpler entry: HTF trend + ADX + RSI asymmetric (no Donchian, no HMA cross)
5. Wider RSI thresholds to generate more trades (target 40-60/year on 4h)

Why this might beat Sharpe=0.520:
- KAMA efficiency ratio adapts to market regime automatically
- ADX > 20 catches trends early (vs >25 which is late)
- Asymmetric RSI generates more signals than pullback zones
- 1d HTF keeps us on right side of major moves
- Fewer filters = more trades while maintaining quality

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 40-60 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_1d_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio.
    ER = |net change| / sum of absolute changes
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    net_change = np.abs(close_s.diff(period))
    total_change = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    er = net_change / (total_change + 1e-10)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    Measures trend strength regardless of direction.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed DM
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for primary trend direction
    kama_1d = calculate_kama(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (Price vs KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # 1d KAMA slope (3 bars)
        kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-3] if i >= 3 else False
        kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-3] if i >= 3 else False
        
        # === 4H KAMA SLOPE ===
        kama_4h_slope_bull = kama_4h[i] > kama_4h[i-2] if i >= 2 else False
        kama_4h_slope_bear = kama_4h[i] < kama_4h[i-2] if i >= 2 else False
        
        # === ADX TREND STRENGTH ===
        trend_strength = adx_14[i] > 20.0  # Trending market
        
        # === ASYMMETRIC RSI ENTRY ===
        # Long: RSI < 50 in uptrend (buying dips)
        # Short: RSI > 50 in downtrend (selling rallies)
        rsi_long_signal = rsi_14[i] < 50.0
        rsi_short_signal = rsi_14[i] > 50.0
        
        # DI crossover confirmation
        di_bull = plus_di[i] > minus_di[i]
        di_bear = plus_di[i] < minus_di[i]
        
        # === ENTRY LOGIC (Simplified for more trades) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d bull trend + 4h momentum + RSI dip + ADX confirms ---
        # Less strict: need 1d trend + RSI < 50 + (ADX OR DI bull)
        if price_above_kama_1d and kama_1d_slope_bull:
            if rsi_long_signal and kama_4h_slope_bull:
                if trend_strength or di_bull:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 1d bear trend + 4h momentum + RSI rally + ADX confirms ---
        if price_below_kama_1d and kama_1d_slope_bear:
            if rsi_short_signal and kama_4h_slope_bear:
                if trend_strength or di_bear:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless exit conditions met
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
            if price_below_kama_1d and kama_1d_slope_bear:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_kama_1d and kama_1d_slope_bull:
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