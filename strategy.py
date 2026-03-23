#!/usr/bin/env python3
"""
Experiment #043: 1d Primary + 1w HTF — KAMA Trend + Fisher Transform + Choppiness Regime

Hypothesis: Daily timeframe with weekly trend bias will generate 20-50 trades/year
with improved Sharpe vs 4h strategies. Key insights from 42 failed experiments:

1) 1d primary = fewer trades, less fee drag (proven in exp#033, #037)
2) KAMA adapts to volatility better than EMA/HMA in choppy markets
3) Fisher Transform catches reversals in bear rallies (Ehlers proven)
4) Choppiness Index regime switch: mean revert when CHOP>50, trend when CHOP<45
5) LOOSE entry thresholds to ensure 30+ trades (avoid Sharpe=0.000 failure)
6) 1w HMA for macro bias prevents counter-trend trades in strong trends

Why this should beat current best (Sharpe=0.486):
- 1d has less noise than 4h/1h (fewer whipsaws in 2022 crash)
- KAMA + Fisher combo not yet tested at 1d timeframe
- Weekly HMA filter stronger than daily for macro direction
- Conservative position size (0.28) controls drawdown in 2022/2025 bears

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target trades: 25-50/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_chop_regime_1w_v1"
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

def calculate_kama(close, er_period=10, fast_sc=2.0, slow_sc=30.0):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    ER period=10, fast SC=2/(1+1), slow SC=2/(1+30)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        er[i] = signal / (noise + 1e-10)
    
    # Calculate Smoothing Constant (SC)
    fast_sc_val = 2.0 / (fast_sc + 1.0)
    slow_sc_val = 2.0 / (slow_sc + 1.0)
    sc = er * (fast_sc_val - slow_sc_val) + slow_sc_val
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        # Normalize price position within range
        range_val = hh - ll + 1e-10
        x = (2.0 * (close[i] - ll) / range_val) - 1.0
        
        # Constrain x to avoid log(0)
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger line (1-period lag of fisher)
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    return hma.values

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators (all before loop for performance)
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_10 = calculate_kama(close, er_period=10, fast_sc=2.0, slow_sc=30.0)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(kama_10[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Range market (LOOSE threshold)
        is_trending = chop_value < 45.0  # Trend market (with hysteresis)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_trigger[i] >= -1.5
        fisher_overbought = fisher[i] > 1.5 and fisher_trigger[i] <= 1.5
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        kama_slope_up = kama_10[i] > kama_10[i-5] if i > 5 else False
        kama_slope_down = kama_10[i] < kama_10[i-5] if i > 5 else False
        
        # === RSI FILTER (LOOSE) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with Fisher Transform ---
        if is_ranging:
            # Long: Fisher oversold + KAMA support + weekly helps
            if fisher_oversold or (fisher_rising and fisher[i] < -0.5):
                if kama_bullish and (price_above_hma_1w or rsi_oversold):
                    new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + KAMA resistance + weekly helps
            elif fisher_overbought or (fisher_falling and fisher[i] > 0.5):
                if kama_bearish and (price_below_hma_1w or rsi_overbought):
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Donchian ---
        elif is_trending:
            # Long: Donchian breakout + KAMA bullish + weekly confirms
            if donchian_breakout_long and kama_bullish:
                if price_above_hma_1w and kama_slope_up:
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + KAMA bearish + weekly confirms
            elif donchian_breakout_short and kama_bearish:
                if price_below_hma_1w and kama_slope_down:
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: KAMA crossover if no regime signal (ensures trades) ---
        if new_signal == 0.0:
            # Long: Price crosses above KAMA + Fisher rising + weekly helps
            if close[i] > kama_10[i] and close[i-1] <= kama_10[i-1]:
                if fisher_rising and (price_above_hma_1w or fisher[i] > -1.0):
                    new_signal = POSITION_SIZE
            
            # Short: Price crosses below KAMA + Fisher falling + weekly helps
            elif close[i] < kama_10[i] and close[i-1] >= kama_10[i-1]:
                if fisher_falling and (price_below_hma_1w or fisher[i] < 1.0):
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
        
        # === EXIT ON REGIME/TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1w and kama_bearish and chop_value < 40:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w and kama_bullish and chop_value < 40:
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