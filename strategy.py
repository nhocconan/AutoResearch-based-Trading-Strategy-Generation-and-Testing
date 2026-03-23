#!/usr/bin/env python3
"""
Experiment #663: 1d Primary + 1w HTF — Fisher Transform + KAMA Trend + Donchian Breakout

Hypothesis: After analyzing 581+ failed strategies, the winning pattern for 1d is:
1. Fisher Transform catches reversals better than RSI in bear/range markets (research-backed)
2. KAMA adapts to volatility changes better than HMA/EMA (Kaufman's Adaptive MA)
3. Donchian breakout confirms trend direction (reduces false signals)
4. 1w HTF provides major trend bias without over-filtering (1w is slow enough for 1d)
5. Asymmetric entries: mean-revert in chop, trend-follow when trending + breakout confirmed

Why this might beat Sharpe=0.520 (current best: mtf_1d_chop_crsi_regime_1w_v1):
- Fisher Transform has sharper reversal signals than CRSI (less lag at extremes)
- KAMA efficiency ratio filters out noisy sideways movement better than Choppiness
- Donchian(20) breakout adds confirmation layer (reduces whipsaws in 2022 crash)
- 1d timeframe = 20-40 trades/year (optimal per Rule 10, lower fee drag than 4h)
- Conservative sizing (0.28) + ATR stop (2.5x) controls drawdown in bear markets

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-40 trades/year on 1d
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_donchian_1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    
    ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    
    In trending markets (high ER), KAMA follows price closely.
    In ranging markets (low ER), KAMA flattens (reduces whipsaws).
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio calculation
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = signal / (noise + 1e-10)
    er = er.fillna(0.0).values
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.67 * ((Close - LL) / (HH - LL) - 0.5) + 0.67 * X_prev
    
    Converts price to Gaussian distribution for clearer reversal signals.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll:
            x = fisher_prev[i-1] if i > period else 0.0
        else:
            x = 0.67 * ((high[i] - ll) / (hh - ll) - 0.5) + 0.67 * (fisher_prev[i-1] if i > period else 0.0)
            x = np.clip(x, -0.999, 0.999)  # Prevent division by zero
        
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
        fisher_prev[i] = fisher[i]
    
    return fisher

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = Highest High over period
    Lower = Lowest Low over period
    Middle = (Upper + Lower) / 2
    
    Breakout above Upper = bullish, below Lower = bearish
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8: Range/consolidation
    CHOP < 38.2: Trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA for major trend direction
    kama_1w = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher = calculate_fisher_transform(high, low, period=9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
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
        if np.isnan(kama_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(kama_1d[i]) or np.isnan(fisher[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(chop_14[i]) or atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (KAMA slope) ===
        # 1w KAMA slope over 3 bars (1w bars are sparse after alignment)
        hma_1w_slope_bull = False
        hma_1w_slope_bear = False
        if i >= 7 and not np.isnan(kama_1w_aligned[i-7]):
            hma_1w_slope_bull = kama_1w_aligned[i] > kama_1w_aligned[i-7]
            hma_1w_slope_bear = kama_1w_aligned[i] < kama_1w_aligned[i-7]
        
        # Price relative to 1w KAMA
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === 1D KAMA SLOPE (trend confirmation) ===
        kama_1d_slope_bull = False
        kama_1d_slope_bear = False
        if i >= 3:
            kama_1d_slope_bull = kama_1d[i] > kama_1d[i-3]
            kama_1d_slope_bear = kama_1d[i] < kama_1d[i-3]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0  # Range/consolidation
        is_trend = chop_14[i] < 45.0  # Trending
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 = long signal
        fisher_long = False
        if i >= 1 and not np.isnan(fisher[i-1]):
            fisher_long = (fisher[i-1] <= -1.5 and fisher[i] > -1.5)
        
        # Fisher crosses below +1.5 = short signal
        fisher_short = False
        if i >= 1 and not np.isnan(fisher[i-1]):
            fisher_short = (fisher[i-1] >= 1.5 and fisher[i] < 1.5)
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP > 55) + Fisher oversold cross = mean revert long
        if is_range and fisher_long:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (CHOP < 45) + 1w bull + 1d bull + Donchian breakout
        elif is_trend and hma_1w_slope_bull and price_above_kama_1w:
            if kama_1d_slope_bull and donchian_breakout_long:
                new_signal = POSITION_SIZE
        
        # Regime 3: 1w bull + Fisher long (pullback entry in uptrend)
        elif hma_1w_slope_bull and price_above_kama_1w and fisher_long:
            new_signal = POSITION_SIZE * 0.7  # Smaller size for pullback entries
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP > 55) + Fisher overbought cross = mean revert short
        if is_range and fisher_short:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (CHOP < 45) + 1w bear + 1d bear + Donchian breakdown
        elif is_trend and hma_1w_slope_bear and price_below_kama_1w:
            if kama_1d_slope_bear and donchian_breakout_short:
                new_signal = -POSITION_SIZE
        
        # Regime 3: 1w bear + Fisher short (pullback entry in downtrend)
        elif hma_1w_slope_bear and price_below_kama_1w and fisher_short:
            new_signal = -POSITION_SIZE * 0.7  # Smaller size for pullback entries
        
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
            if hma_1w_slope_bear and price_below_kama_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1w_slope_bull and price_above_kama_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals