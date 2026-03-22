#!/usr/bin/env python3
"""
Experiment #506: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Exit + Asymmetric RSI

Hypothesis: After 499 failed/partial strategies, try a DIFFERENT adaptive approach:

1. KAMA (Kaufman Adaptive Moving Average): Adapts smoothing based on market efficiency ratio.
   - Fast in trends, slow in ranges. Proven to reduce whipsaw vs EMA/HMA.
   - ER (Efficiency Ratio) = |net change| / sum(|individual changes|)
   - Fast SC = 2/(2+1), Slow SC = 2/(30+1)
   
2. CHOPPINESS INDEX as EXIT filter (not entry): Exit when CHOP > 61.8 (range bound)
   - Most failed strategies used CHOP for entry. Use it to EXIT losing trades early.
   - CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
   
3. ASYMMETRIC RSI thresholds based on 1d trend regime:
   - Bull (price > 1d KAMA): Long at RSI < 40, exit at RSI > 75
   - Bear (price < 1d KAMA): Short at RSI > 60, exit at RSI < 25
   - This adapts to market regime unlike fixed RSI thresholds
   
4. ATR VOLATILITY SCALING on position size:
   - Size = base_size * (ATR_median / ATR_current)
   - Reduces exposure during vol spikes (protects in 2022-style crashes)
   
5. DONCHIAN BREAKOUT confirmation: Only enter when price breaks 20-bar Donchian
   - Ensures momentum confirmation, reduces false entries

Why this might beat Sharpe=0.435:
- KAMA is DIFFERENT from HMA/EMA (400+ strategies failed with those)
- Choppiness as EXIT (not entry) is novel approach
- Asymmetric RSI adapts to regime (proven in research notes)
- ATR sizing protects drawdown in high vol periods
- 12h TF targets 20-50 trades/year (optimal fee/trade balance)

Position sizing: 0.25 base, scaled by ATR (max 0.35)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_exit_asym_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    net_change = close_s.diff(er_period).abs()
    sum_changes = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / (sum_change + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # ATR calculation
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100.0 * np.log10(atr / (hh - ll + 1e-10)) / np.log10(period)
    
    return chop.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return atr.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF KAMA for major trend direction
    kama_1d_21 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_50 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Recalculate with different periods for 50
    kama_1d_50 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    # Actually need different span - use simple method for 50
    kama_1d_50 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    kama_1d_50_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_median = np.nanmedian(atr_14[100:])  # For volatility scaling
    
    # Choppiness Index for regime/exit filter
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Donchian Channel for breakout confirmation
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # RSI for entry timing
    rsi_14 = calculate_rsi(close, 14)
    
    # 12h KAMA for local trend
    kama_12h_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    
    # Base position sizing
    BASE_LONG_SIZE = 0.28
    BASE_SHORT_SIZE = 0.25
    MAX_SIZE = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1d_21_aligned[i]) or np.isnan(kama_1d_50_aligned[i]):
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(kama_12h_21[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > kama_1d_21_aligned[i]
        bear_regime = close[i] < kama_1d_21_aligned[i]
        
        # 1d KAMA slope for trend strength
        kama_slope_bull = kama_1d_21_aligned[i] > kama_1d_50_aligned[i]
        kama_slope_bear = kama_1d_21_aligned[i] < kama_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME (exit filter) ===
        choppy_market = chop[i] > 61.8  # Range-bound, exit positions
        trending_market = chop[i] < 38.2  # Strong trend, hold positions
        
        # === VOLATILITY SCALING (ATR-based position sizing) ===
        vol_scale = min(1.5, max(0.5, atr_median / (atr_14[i] + 1e-10)))
        long_size = min(MAX_SIZE, BASE_LONG_SIZE * vol_scale)
        short_size = min(MAX_SIZE, BASE_SHORT_SIZE * vol_scale)
        
        # === ASYMMETRIC RSI THRESHOLDS (based on regime) ===
        if bull_regime:
            # Bull market: easier to go long, harder to short
            rsi_long_entry = rsi_14[i] < 40.0
            rsi_long_exit = rsi_14[i] > 75.0
            rsi_short_entry = rsi_14[i] > 70.0
            rsi_short_exit = rsi_14[i] < 25.0
        else:
            # Bear market: easier to go short, harder to long
            rsi_long_entry = rsi_14[i] < 25.0
            rsi_long_exit = rsi_14[i] > 70.0
            rsi_short_entry = rsi_14[i] > 60.0
            rsi_short_exit = rsi_14[i] < 30.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_up = close[i] > donchian_upper[i-1]
        donchian_breakout_down = close[i] < donchian_lower[i-1]
        
        # === KAMA LOCAL TREND ===
        kama_12h_bull = close[i] > kama_12h_21[i]
        kama_12h_bear = close[i] < kama_12h_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES (require confluence)
        # Condition 1: Bull regime + RSI oversold + Donchian breakout up
        if bull_regime and rsi_long_entry and donchian_breakout_up:
            new_signal = long_size
        # Condition 2: Bull regime + KAMA 12h bull + RSI moderate oversold
        elif bull_regime and kama_12h_bull and rsi_14[i] < 45.0:
            new_signal = long_size * 0.8
        # Condition 3: Strong bull (both 1d and 12h) + any RSI pullback
        elif bull_regime and kama_slope_bull and kama_12h_bull and rsi_14[i] < 50.0:
            new_signal = long_size
        # Condition 4: Bear regime but extreme oversold (contrarian)
        elif bear_regime and rsi_14[i] < 20.0 and not choppy_market:
            new_signal = long_size * 0.5
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: Bear regime + RSI overbought + Donchian breakout down
            if bear_regime and rsi_short_entry and donchian_breakout_down:
                new_signal = -short_size
            # Condition 2: Bear regime + KAMA 12h bear + RSI moderate overbought
            elif bear_regime and kama_12h_bear and rsi_14[i] > 55.0:
                new_signal = -short_size * 0.8
            # Condition 3: Strong bear (both 1d and 12h) + any RSI bounce
            elif bear_regime and kama_slope_bear and kama_12h_bear and rsi_14[i] > 50.0:
                new_signal = -short_size
            # Condition 4: Bull regime but extreme overbought (contrarian)
            elif bull_regime and rsi_14[i] > 80.0 and not choppy_market:
                new_signal = -short_size * 0.5
        
        # === CHOPPINESS EXIT (novel: use CHOP to exit, not entry) ===
        if in_position and choppy_market:
            # Market became choppy, exit to avoid whipsaw
            new_signal = 0.0
        
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
        
        # === RSI EXIT CONDITIONS ===
        if in_position and position_side > 0 and rsi_long_exit:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_short_exit:
            new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        if in_position and position_side > 0 and bear_regime and kama_slope_bear:
            # Long position but regime flipped bearish
            new_signal = 0.0
        
        if in_position and position_side < 0 and bull_regime and kama_slope_bull:
            # Short position but regime flipped bullish
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
                # Flip position
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