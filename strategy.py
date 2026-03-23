#!/usr/bin/env python3
"""
Experiment #031: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Reversal + Choppiness Regime

Hypothesis: 4h timeframe with adaptive KAMA trend filter + Ehlers Fisher Transform for reversals
will outperform static EMA/HMA strategies. KAMA adapts to volatility (less whipsaw in chop),
Fisher Transform catches turning points better than RSI, and Choppiness regime filter prevents
trend strategies in ranging markets. 1d/1w HTF provides macro bias for higher win rate.

Key innovations vs failed experiments:
1. KAMA (Kaufman Adaptive) instead of HMA/EMA — adapts efficiency ratio to market conditions
2. Fisher Transform (period=9) for reversal signals — superior to RSI at extremes
3. Asymmetric position sizing: 0.35 in trend regime, 0.25 in range regime
4. Dual HTF confirmation: 1d for intermediate trend, 1w for macro bias
5. LOOSE entries to ensure trade generation (Fisher > -1.5 or < +1.5, not extreme values)

Why 4h works: Proven in exp#021 (Sharpe 0.486). Targets 20-50 trades/year = 80-200 over 4yr train.
Position sizing: 0.25-0.35 discrete. Stoploss: 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_regime_1d1w_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency — smooth in chop, responsive in trends.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER): |net change| / sum of absolute changes
    net_change = np.abs(close - np.roll(close, period))
    net_change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = net_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    Entry: Fisher crosses above -1.5 (long) or below +1.5 (short)
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price position within range
        range_val = hh - ll
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
        else:
            price_position = (high[i] + low[i]) / 2.0
            normalized = (price_position - ll) / range_val
            normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
            fisher[i] = 0.67 * fisher[i] + 0.33 * (fisher[i-1] if i > 0 else 0.0)
        
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) for HTF trend."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_10 = calculate_kama(close, period=10)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # KAMA slope (5-bar lookback)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        kama_slope[i] = kama_10[i] - kama_10[i-5]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.35  # Larger in trending regime
    SIZE_RANGE = 0.25  # Smaller in ranging regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_10[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or atr_14[i] == 0:
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both 1d and 1w aligned
        strong_bullish = price_above_hma_1d and price_above_hma_1w
        strong_bearish = price_below_hma_1d and price_below_hma_1w
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market (mean revert)
        is_trending = chop_value < 45.0  # Trend market (trend follow)
        # Hysteresis zone: 45-55 = hold previous regime
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        kama_slope_up = kama_slope[i] > 0
        kama_slope_down = kama_slope[i] < 0
        
        # === FISHER TRANSFORM REVERSAL ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === RSI CONFIRMATION (LOOSE for trade generation) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        current_size = SIZE_RANGE  # Default to smaller size
        
        # --- TRENDING REGIME: KAMA Trend Following ---
        if is_trending:
            current_size = SIZE_TREND
            
            # Long: KAMA bullish + slope up + Fisher confirmation + HTF bias
            if kama_bullish and kama_slope_up:
                if fisher_cross_up or (fisher_oversold and rsi_rising):
                    if strong_bullish or price_above_hma_1d:
                        new_signal = current_size
            
            # Short: KAMA bearish + slope down + Fisher confirmation + HTF bias
            elif kama_bearish and kama_slope_down:
                if fisher_cross_down or (fisher_overbought and rsi_falling):
                    if strong_bearish or price_below_hma_1d:
                        new_signal = -current_size
        
        # --- RANGING REGIME: Fisher Mean Reversion ---
        elif is_ranging:
            current_size = SIZE_RANGE
            
            # Long: Fisher oversold + RSI oversold + price near support
            if fisher_oversold and rsi_oversold:
                if rsi_rising or fisher[i] > fisher_signal[i]:
                    # Only long if not strongly bearish on HTF
                    if not strong_bearish:
                        new_signal = current_size
            
            # Short: Fisher overbought + RSI overbought + price near resistance
            elif fisher_overbought and rsi_overbought:
                if rsi_falling or fisher[i] < fisher_signal[i]:
                    # Only short if not strongly bullish on HTF
                    if not strong_bullish:
                        new_signal = -current_size
        
        # --- NEUTRAL REGIME (45-55 CHOP): KAMA crossover only ---
        else:
            # Long: Price crosses above KAMA + RSI rising
            if close[i] > kama_10[i] and close[i-1] <= kama_10[i-1]:
                if rsi_rising and not strong_bearish:
                    new_signal = SIZE_RANGE
            
            # Short: Price crosses below KAMA + RSI falling
            elif close[i] < kama_10[i] and close[i-1] >= kama_10[i-1]:
                if rsi_falling and not strong_bullish:
                    new_signal = -SIZE_RANGE
        
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
        
        # === EXIT ON HTF TREND REVERSAL ===
        # Exit long if 1w turns strongly bearish
        if in_position and position_side > 0:
            if strong_bearish and chop_value < 40:
                new_signal = 0.0
        
        # Exit short if 1w turns strongly bullish
        if in_position and position_side < 0:
            if strong_bullish and chop_value < 40:
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