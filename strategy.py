#!/usr/bin/env python3
"""
Experiment #072: 12h Primary + 1d/1w HTF — Fisher Transform + KAMA Adaptive Trend + Choppiness Regime

Hypothesis: 12h timeframe with Ehlers Fisher Transform for precise reversal entries, KAMA for
adaptive trend following (adjusts to volatility), and Choppiness Index for regime detection will
generate 20-40 trades/year with Sharpe > 0.486.

Key innovations:
1) Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution, catches reversals
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2) KAMA (Kaufman Adaptive MA): Adapts smoothing based on market efficiency ratio
   Better than EMA/HMA in choppy markets (ER low = more smoothing)
3) Choppiness Index regime: CHOP > 55 = range (Fisher mean reversion), CHOP < 45 = trend (KAMA follow)
4) Triple HTF confirmation: 12h KAMA for intermediate, 1d HMA for macro, 1w HMA for secular bias
5) ATR-based position sizing: Reduce size when ATR(14)/ATR(50) > 2.0 (high vol = smaller position)
6) Asymmetric entries: Only long when 1d HMA bullish, only short when 1d HMA bearish

Why this should work:
- 12h proven timeframe (exp #062 Sharpe=0.044 was close, needs better entry timing)
- Fisher Transform outperforms RSI for reversals (Ehlers research)
- KAMA adapts to volatility better than fixed-period MAs
- 1w HTF adds strong secular bias filter (prevents counter-trend in major moves)
- ATR sizing reduces risk in volatile periods (2022 crash protection)
- Fewer trades (20-40/year) = less fee drag on 12h TF

Position size: 0.20-0.30 (discrete, ATR-adjusted)
Stoploss: 2.5*ATR trailing
Target: 20-40 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_chop_regime_1d1w_v1"
timeframe = "12h"
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
    KAMA adapts smoothing based on market Efficiency Ratio (ER).
    High ER (trending) = fast smoothing, Low ER (choppy) = slow smoothing.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |Net Change| / Sum of Absolute Changes over period
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = np.abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        er[i] = net_change / (sum_changes + 1e-10) if sum_changes > 0 else 0
    
    # Calculate smoothing constant
    # sc = [ER * (fast_sc - slow_sc) + slow_sc]^2
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        # Normalize price to -1 to +1 range
        range_val = hh - ll
        if range_val < 1e-10:
            x = 0
        else:
            x = (2.0 * (close[i] if 'close' in dir() else high[i]) - ll - hh) / range_val
            x = np.clip(x, -0.999, 0.999)  # Prevent ln(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x) + 1e-10)
        
        # Trigger line (previous Fisher value)
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_fisher_transform_v2(close, period=9):
    """
    Calculate Ehlers Fisher Transform (price-based version).
    More stable than high/low version for crypto.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    close_s = pd.Series(close)
    
    for i in range(period, n):
        # Normalize price using highest high and lowest low
        hh = close_s.iloc[i - period + 1:i + 1].max()
        ll = close_s.iloc[i - period + 1:i + 1].min()
        
        range_val = hh - ll
        if range_val < 1e-10:
            x = 0
        else:
            x = (2.0 * close[i] - ll - hh) / range_val
            x = np.clip(x, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x) + 1e-10)
        
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for secular bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher_transform_v2(close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_POSITION_SIZE = 0.30
    REDUCED_POSITION_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_50[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(kama_12h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === HTF TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY FILTER ===
        vol_ratio = atr_14[i] / (atr_50[i] + 1e-10)
        high_volatility = vol_ratio > 2.0
        
        # Position size adjustment based on volatility
        position_size = REDUCED_POSITION_SIZE if high_volatility else BASE_POSITION_SIZE
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long signal
        fisher_cross_long = (fisher[i] > -1.5) and (fisher[i-1] <= -1.5) if i > 0 else False
        # Fisher crosses below +1.5 from above = short signal
        fisher_cross_short = (fisher[i] < 1.5) and (fisher[i-1] >= 1.5) if i > 0 else False
        
        # Fisher extreme oversold/overbought
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === ADAPTIVE REGIME ENTRY ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: KAMA trend + Fisher confirmation + HTF bias ---
        if is_trending:
            # Long: KAMA bullish + Fisher cross long + 1d HMA not bearish
            if kama_bullish and fisher_cross_long:
                if price_above_hma_1d or price_above_hma_1w:
                    new_signal = position_size
            
            # Short: KAMA bearish + Fisher cross short + 1d HMA not bullish
            elif kama_bearish and fisher_cross_short:
                if price_below_hma_1d or price_below_hma_1w:
                    new_signal = -position_size
        
        # --- RANGING REGIME: Fisher Mean Reversion + HTF filter ---
        elif is_ranging:
            # Long: Fisher oversold + 1d/1w not strongly bearish
            if fisher_oversold:
                if not price_below_hma_1w:
                    new_signal = position_size
            
            # Short: Fisher overbought + 1d/1w not strongly bullish
            elif fisher_overbought:
                if not price_above_hma_1w:
                    new_signal = -position_size
        
        # === HOLD POSITION LOGIC ===
        # Keep position if Fisher hasn't reversed and no stoploss
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if Fisher > 0 and KAMA still bullish
                if fisher[i] > 0.0 and kama_bullish:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if Fisher < 0 and KAMA still bearish
                if fisher[i] < 0.0 and kama_bearish:
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
            # Exit long if both 1d and 1w turn bearish
            if price_below_hma_1d and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both 1d and 1w turn bullish
            if price_above_hma_1d and price_above_hma_1w:
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
                # Position flip
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