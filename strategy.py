#!/usr/bin/env python3
"""
Experiment #052: 12h Primary + 1d/1w HTF — Fisher Transform + KAMA Trend + Donchian Breakout

Hypothesis: 12h timeframe with daily/weekly trend bias using Fisher Transform (proven reversal
indicator) + KAMA adaptive trend + Donchian breakout will generate 20-50 trades/year with Sharpe > 0.486.

Key insights from 46+ failed experiments:
1) CRSI+Choppiness has failed 5+ times — need DIFFERENT approach
2) Fisher Transform showed promise in #047 (1d KAMA+Fisher+Chop kept with Sharpe=0.141)
3) KAMA adapts to market efficiency — works better than EMA in choppy markets
4) Donchian breakout confirms trend direction without lag
5) 12h primary = proven higher TF works best (target 20-50 trades/year)
6) Dual HTF (1d + 1w) provides strong macro bias without over-filtering

Why this should work:
- Fisher Transform catches reversals at extremes (better than RSI for timing)
- KAMA adapts speed based on market regime (fast in trends, slow in chop)
- Donchian(20) breakout confirms momentum direction
- 1d/1w HMA provides ultra-long-term bias (prevents counter-trend trades)
- 12h timeframe = fewer trades, less fee drag, cleaner signals

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 20-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_donchian_1d1w_v1"
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
    KAMA adapts speed based on market efficiency ratio.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = np.abs(close[0:er_period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=er_period, min_periods=er_period).sum().values
    volatility[0:er_period] = np.abs(close[0:er_period] - close[0])
    
    er = change / (volatility + 1e-10)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constant
    sc = (er * (2.0/(fast_period+1) - 2.0/(slow_period+1)) + 2.0/(slow_period+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        # Normalize price
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        normalized = (2.0 * (close[i] - lowest) / range_val) - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Trigger line (previous fisher)
        trigger[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, trigger

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and DX
    plus_di = 100.0 * plus_smooth / (tr_smooth + 1e-10)
    minus_di = 100.0 * minus_smooth / (tr_smooth + 1e-10)
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for medium-term bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for long-term bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_14 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, trigger = calculate_fisher(close, period=9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(donchian_upper[i]) or np.isnan(adx[i]):
            continue
        
        # === MACRO BIAS (1d + 1w) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bullish: both 1d and 1w above HMA
        macro_bullish = price_above_hma_1d and price_above_hma_1w
        # Strong bearish: both 1d and 1w below HMA
        macro_bearish = price_below_hma_1d and price_below_hma_1w
        # Neutral: mixed signals
        macro_neutral = not macro_bullish and not macro_bearish
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_14[i]
        kama_bearish = close[i] < kama_14[i]
        kama_slope_up = kama_14[i] > kama_14[i-5] if i > 5 else False
        kama_slope_down = kama_14[i] < kama_14[i-5] if i > 5 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and trigger[i] > fisher[i]  # Crossing up from oversold
        fisher_overbought = fisher[i] > 1.5 and trigger[i] < fisher[i]  # Crossing down from overbought
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 25.0
        adx_weak = adx[i] < 20.0
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- STRONG TREND REGIME (ADX > 25 + macro bias) ---
        if adx_strong:
            # Long: KAMA bullish + Fisher not overbought + Donchian breakout + macro confirms
            if kama_bullish and di_bullish and not fisher_overbought:
                if macro_bullish or donchian_breakout_up:
                    new_signal = POSITION_SIZE
            
            # Short: KAMA bearish + Fisher not oversold + Donchian breakdown + macro confirms
            elif kama_bearish and di_bearish and not fisher_oversold:
                if macro_bearish or donchian_breakout_down:
                    new_signal = -POSITION_SIZE
        
        # --- WEAK TREND / RANGE REGIME (ADX < 20) ---
        elif adx_weak:
            # Mean reversion: Fisher extremes with macro filter
            if fisher_oversold and macro_bullish:
                new_signal = POSITION_SIZE
            elif fisher_overbought and macro_bearish:
                new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME (ADX 20-25) ---
        else:
            # KAMA trend with Fisher confirmation
            if kama_bullish and fisher_rising and not fisher_overbought:
                if macro_bullish or macro_neutral:
                    new_signal = POSITION_SIZE
            elif kama_bearish and fisher_falling and not fisher_oversold:
                if macro_bearish or macro_neutral:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if Fisher not at opposite extreme
            if position_side > 0 and fisher[i] < 1.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and fisher[i] > -1.0:
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
        
        # === EXIT ON MACRO TREND CHANGE ===
        if in_position and position_side > 0:
            if macro_bearish and kama_bearish:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bullish and kama_bullish:
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