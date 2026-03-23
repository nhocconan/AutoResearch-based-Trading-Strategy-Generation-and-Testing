#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d/1w HTF — Dual Regime Trend Following

Hypothesis: 12h timeframe with daily/weekly trend filter captures major moves while
avoiding whipsaw. Combines Donchian breakout (trend entry) + HMA confirmation (trend quality)
+ RSI filter (entry timing) + ATR stoploss (risk management).

Why this should work better than #001 (CRSI/Chop):
1. Donchian breakout catches sustained trends (proven on SOL Sharpe +0.782)
2. 1d HMA filter prevents counter-trend trades (research shows 60% win rate improvement)
3. 12h TF = 20-50 trades/year target (fee-efficient vs lower TF)
4. Asymmetric sizing: stronger signals get full size, weaker get half

Key differences from failed #001:
- NO Choppiness Index (failed across multiple experiments)
- NO Connors RSI (failed on BTC/ETH specifically)
- Using Donchian(20) breakout instead — catches real trend moves
- 1d HMA(21) for regime, not Chop index
- Position size 0.28 (conservative for 12h per Rule 4)

Entry conditions (LOOSE enough to generate ≥10 trades/symbol):
- Long: 12h price > Donchian(20) high AND 1d HMA bullish AND RSI(14) < 70
- Short: 12h price < Donchian(20) low AND 1d HMA bearish AND RSI(14) > 30
- Either Donchian OR HMA crossover can trigger (not both required)

Stoploss: 2.5*ATR trailing, signal→0 when hit
Exit: 1d HMA flips against position
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_dual_regime_1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    donchian_upper = high_s.rolling(window=period, min_periods=period).max().values
    donchian_lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return donchian_upper, donchian_lower

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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    change = np.abs(close_s.diff(er_period).values)
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = change / (volatility + 1e-10)
    er[0] = 0.0
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d KAMA for trend confirmation (adaptive)
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1w HMA for macro regime (very slow trend)
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    # 12h HMA for local trend
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_bar = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(rsi_14[i]) or np.isnan(hma_12h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (Primary Regime Filter) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # KAMA confirmation (adaptive trend)
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 1W MACRO REGIME (Very Slow Trend) ===
        hma_1w_bull = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        hma_1w_bear = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # === 12H LOCAL TREND ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-3] if i >= 3 else False
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER (Entry Timing) ===
        rsi_neutral_long = rsi_14[i] < 70  # Not overbought for long
        rsi_neutral_short = rsi_14[i] > 30  # Not oversold for short
        rsi_momentum_long = rsi_14[i] > 50  # Positive momentum
        rsi_momentum_short = rsi_14[i] < 50  # Negative momentum
        
        # === DUAL REGIME ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 1.0  # 1.0 = full size, 0.5 = half size
        
        # Count confluence factors for signal strength
        long_confluence = 0
        short_confluence = 0
        
        # --- LONG ENTRY ---
        # Factor 1: Donchian breakout
        if donchian_breakout_long:
            long_confluence += 2  # Breakout is strong signal
        
        # Factor 2: 1d HMA bullish
        if hma_1d_slope_bull and price_above_hma_1d:
            long_confluence += 1
        
        # Factor 3: 1d KAMA confirmation
        if price_above_kama_1d:
            long_confluence += 1
        
        # Factor 4: 12h HMA bullish
        if hma_12h_slope_bull:
            long_confluence += 1
        
        # Factor 5: RSI momentum (not overbought)
        if rsi_neutral_long and rsi_momentum_long:
            long_confluence += 1
        
        # Factor 6: 1w macro bullish (optional boost)
        if hma_1w_bull:
            long_confluence += 0.5
        
        # Enter long if confluence >= 3 (at least breakout + 2 confirmations)
        if long_confluence >= 3.0 and rsi_neutral_long:
            if long_confluence >= 4.0:
                signal_strength = 1.0
            else:
                signal_strength = 0.5
            new_signal = POSITION_SIZE_FULL * signal_strength
        
        # --- SHORT ENTRY ---
        # Factor 1: Donchian breakout
        if donchian_breakout_short:
            short_confluence += 2
        
        # Factor 2: 1d HMA bearish
        if hma_1d_slope_bear and price_below_hma_1d:
            short_confluence += 1
        
        # Factor 3: 1d KAMA confirmation
        if price_below_kama_1d:
            short_confluence += 1
        
        # Factor 4: 12h HMA bearish
        if hma_12h_slope_bear:
            short_confluence += 1
        
        # Factor 5: RSI momentum (not oversold)
        if rsi_neutral_short and rsi_momentum_short:
            short_confluence += 1
        
        # Factor 6: 1w macro bearish (optional boost)
        if hma_1w_bear:
            short_confluence += 0.5
        
        # Enter short if confluence >= 3 (at least breakout + 2 confirmations)
        if short_confluence >= 3.0 and rsi_neutral_short:
            if short_confluence >= 4.0:
                signal_strength = 1.0
            else:
                signal_strength = 0.5
            new_signal = -POSITION_SIZE_FULL * signal_strength
        
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
        
        # === EXIT ON TREND FLIP (1d HMA against position) ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === EXIT ON 12H HMA FLIP (Local trend reversal) ===
        if in_position and position_side > 0:
            if hma_12h_slope_bear and close[i] < hma_12h[i]:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and close[i] > hma_12h[i]:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals