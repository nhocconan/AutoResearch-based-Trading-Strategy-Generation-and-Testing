#!/usr/bin/env python3
"""
Experiment #034: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Ehlers Fisher Transform

Hypothesis: KAMA (Kaufman Adaptive Moving Average) automatically adjusts to market efficiency,
working well in both trending and ranging markets. Combined with Ehlers Fisher Transform
for reversal detection and volume confirmation for conviction, this should outperform
simple EMA/HMA strategies that failed on BTC/ETH.

Key innovations:
1. KAMA(10,2,30): Adapts smoothing based on market noise ratio - less lag in trends, more smoothing in chop
2. Ehlers Fisher Transform(9): Normalizes price to -1/+1 range, extreme crossings signal reversals
3. Volume confirmation: taker_buy_volume ratio confirms institutional participation
4. 12h KAMA slope for intermediate trend bias
5. 1d KAMA for macro directional filter
6. Choppiness regime switch: mean-revert in chop, trend-follow otherwise
7. ATR trailing stop (2.5*ATR) with signal→0 on stop

Why this should work:
- KAMA outperforms EMA/HMA in adaptive markets (research-validated)
- Fisher Transform catches reversals earlier than RSI/CRSI
- Volume filter reduces false signals
- 4h timeframe targets 30-50 trades/year (fee-efficient)
- Loose enough entries to guarantee ≥10 trades/symbol

Position size: 0.28 (discrete, within 0.20-0.35 range per Rule 4)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_volume_regime_12h1d_v1"
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

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - smooths in chop, follows in trends.
    
    ER (Efficiency Ratio) = |Change| / Sum(|Individual Changes|)
    SC (Smoothing Constant) = [ER * (fast - slow) + slow]^2
    KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    price_change = np.abs(close_s.diff(efficiency_period).values)
    sum_changes = pd.Series(np.abs(close_s.diff().values)).rolling(window=efficiency_period, min_periods=efficiency_period).sum().values
    
    er = price_change / (sum_changes + 1e-10)
    er[0:efficiency_period] = 0.0  # Not enough data
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[efficiency_period] = close[efficiency_period]  # Initialize
    
    for i in range(efficiency_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range for better reversal detection.
    
    Price = 0.66 * ((Close - LL) / (HH - LL) - 0.5) + 0.67 * Prev_Price
    Fisher = 0.5 * ln((1 + Price) / (1 - Price))
    """
    n = len(high)
    close = (high + low) / 2.0  # Typical price
    
    fisher = np.zeros(n)
    price = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        range_hl = hh - ll + 1e-10
        
        # Calculate normalized price
        price[i] = 0.66 * ((close[i] - ll) / range_hl - 0.5) + 0.67 * price[i-1]
        
        # Clamp to avoid log errors
        price[i] = np.clip(price[i], -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + price[i]) / (1.0 - price[i]))
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_ratio(taker_buy_volume, total_volume, period=14):
    """Calculate volume ratio (taker buy / total) with smoothing."""
    ratio = taker_buy_volume / (total_volume + 1e-10)
    ratio_s = pd.Series(ratio)
    smoothed = ratio_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return smoothed

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA for trend bias
    kama_12h = calculate_kama(df_12h['close'].values, efficiency_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1d KAMA for macro bias
    kama_1d = calculate_kama(df_1d['close'].values, efficiency_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    kama_4h = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    fisher = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_4h[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Mean reversion mode
        is_trending = chop_value < 40.0  # Trend following mode
        
        # === 1D MACRO BIAS ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 12H TREND BIAS ===
        kama_12h_slope_bull = kama_12h_aligned[i] > kama_12h_aligned[i-5] if i >= 5 else False
        kama_12h_slope_bear = kama_12h_aligned[i] < kama_12h_aligned[i-5] if i >= 5 else False
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_4h_slope_bull = kama_4h[i] > kama_4h[i-3] if i >= 3 else False
        kama_4h_slope_bear = kama_4h[i] < kama_4h[i-3] if i >= 3 else False
        price_above_kama_4h = close[i] > kama_4h[i]
        price_below_kama_4h = close[i] < kama_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -0.8  # Strong reversal signal
        fisher_overbought = fisher[i] > 0.8  # Strong reversal signal
        fisher_cross_up = fisher[i] > fisher[i-1] and fisher[i-1] <= -0.5 if i >= 1 else False
        fisher_cross_down = fisher[i] < fisher[i-1] and fisher[i-1] >= 0.5 if i >= 1 else False
        
        # === VOLUME CONFIRMATION ===
        vol_bullish = vol_ratio[i] > 0.55  # More buyer aggression
        vol_bearish = vol_ratio[i] < 0.45  # More seller aggression
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with Fisher extremes ---
        if is_ranging:
            # Long: Fisher oversold + volume bullish + macro neutral/bullish
            if fisher_oversold or fisher_cross_up:
                if vol_bullish and (price_above_kama_1d or chop_value > 55):
                    new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + volume bearish + macro neutral/bearish
            elif fisher_overbought or fisher_cross_down:
                if vol_bearish and (price_below_kama_1d or chop_value > 55):
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with KAMA confluence ---
        elif is_trending:
            # Long: KAMA bullish confluence + Fisher not overbought + volume confirm
            if kama_4h_slope_bull and kama_12h_slope_bull:
                if fisher[i] < 0.5 and vol_bullish and price_above_kama_1d:
                    new_signal = POSITION_SIZE
            
            # Short: KAMA bearish confluence + Fisher not oversold + volume confirm
            elif kama_4h_slope_bear and kama_12h_slope_bear:
                if fisher[i] > -0.5 and vol_bearish and price_below_kama_1d:
                    new_signal = -POSITION_SIZE
        
        # --- TRANSITION ZONE (40 < CHOP < 50): Use both signals ---
        else:
            # Long: Fisher reversal OR KAMA trend with volume
            if (fisher_oversold or fisher_cross_up) and vol_bullish:
                new_signal = POSITION_SIZE
            elif (kama_4h_slope_bull and kama_12h_slope_bull) and fisher[i] < 0.3 and vol_bullish:
                new_signal = POSITION_SIZE
            
            # Short: Fisher reversal OR KAMA trend with volume
            elif (fisher_overbought or fisher_cross_down) and vol_bearish:
                new_signal = -POSITION_SIZE
            elif (kama_4h_slope_bear and kama_12h_slope_bear) and fisher[i] > -0.3 and vol_bearish:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime becomes strongly trending bearish
        if in_position and position_side > 0:
            if is_trending and kama_12h_slope_bear and price_below_kama_1d:
                new_signal = 0.0
        
        # Exit short if regime becomes strongly trending bullish
        if in_position and position_side < 0:
            if is_trending and kama_12h_slope_bull and price_above_kama_1d:
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