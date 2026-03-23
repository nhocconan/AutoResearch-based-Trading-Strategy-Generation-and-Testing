#!/usr/bin/env python3
"""
Experiment #701: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Donchian Breakout + HTF Bias

Hypothesis: After 607+ failed strategies, the pattern shows:
1. Pure trend following fails in 2022 crash and 2025 bear market
2. Pure mean reversion fails in strong trends (2021 bull, SOL rallies)
3. KAMA (Kaufman Adaptive) adapts to market efficiency ratio - works in both regimes
4. Donchian breakout provides clear entry signals without overfiltering
5. 1d HMA gives major trend bias, 1w ADX filters extreme chop

Why this might beat Sharpe=0.520:
- KAMA ER (Efficiency Ratio) automatically adjusts sensitivity - no regime switching needed
- Donchian(20) breakout is proven across all market conditions
- 1d HMA(21) slope gives trend bias without being too slow
- 1w ADX prevents entries in extreme chop (ADX<15)
- Asymmetric sizing: 0.30 with trend, 0.20 counter-trend
- Looser entry conditions to ensure trade generation (learned from #685 0-trade failure)

Position sizing: 0.25-0.30 discrete
Target: 25-45 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing

CRITICAL: Entry conditions deliberately loose to ensure trades generate on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_hma1d_adx1w_v1"
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
    
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |Net Change| / Sum of Absolute Changes over period
    High ER = trending (fast SC), Low ER = choppy (slow SC)
    
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (price - KAMA_prev)
    """
    close_s = pd.Series(close)
    
    # Net change over ER period
    net_change = np.abs(close_s - close_s.shift(er_period))
    
    # Sum of absolute changes (volatility)
    abs_changes = np.abs(close_s.diff())
    sum_changes = abs_changes.rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0).clip(0, 1)
    
    # Smoothing Constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2.0
    
    return upper.values, lower.values, mid.values

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for major trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w ADX for extreme chop filter
    adx_1w_high = df_1w['high'].values
    adx_1w_low = df_1w['low'].values
    adx_1w_close = df_1w['close'].values
    adx_1w = calculate_adx(adx_1w_high, adx_1w_low, adx_1w_close, period=14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_WITH_TREND = 0.30
    SIZE_COUNTER_TREND = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(adx_1w_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D HMA TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1W ADX CHOP FILTER ===
        adx_1w_val = adx_1w_aligned[i]
        extreme_chop = adx_1w_val < 15.0  # Don't trade in extreme chop
        
        # === KAMA TREND ===
        kama_slope_bull = kama_4h[i] > kama_4h[i-3] if i >= 3 else False
        kama_slope_bear = kama_4h[i] < kama_4h[i-3] if i >= 3 else False
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === RSI FILTER (loose to ensure trades) ===
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        
        # === ENTRY LOGIC (deliberately loose for trade generation) ===
        new_signal = 0.0
        
        # Skip if extreme chop on weekly
        if not extreme_chop:
            # --- LONG ENTRY ---
            # Condition 1: Donchian breakout up + KAMA bull + RSI not overbought
            if donchian_breakout_up and kama_slope_bull and rsi_oversold:
                if price_above_hma_1d or hma_1d_slope_bull:
                    new_signal = SIZE_WITH_TREND
                else:
                    new_signal = SIZE_COUNTER_TREND
            
            # Condition 2: Price above KAMA + RSI pullback (mean reversion in trend)
            elif price_above_kama and kama_slope_bull and rsi_14[i] < 40.0:
                if price_above_hma_1d:
                    new_signal = SIZE_WITH_TREND
                else:
                    new_signal = SIZE_COUNTER_TREND
            
            # --- SHORT ENTRY ---
            # Condition 1: Donchian breakout down + KAMA bear + RSI not oversold
            if donchian_breakout_down and kama_slope_bear and rsi_overbought:
                if price_below_hma_1d or hma_1d_slope_bear:
                    new_signal = -SIZE_WITH_TREND
                else:
                    new_signal = -SIZE_COUNTER_TREND
            
            # Condition 2: Price below KAMA + RSI rally (mean reversion in downtrend)
            elif price_below_kama and kama_slope_bear and rsi_14[i] > 60.0:
                if price_below_hma_1d:
                    new_signal = -SIZE_WITH_TREND
                else:
                    new_signal = -SIZE_COUNTER_TREND
        
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
            if kama_slope_bear and price_below_kama:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_slope_bull and price_above_kama:
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