#!/usr/bin/env python3
"""
Experiment #009: 4h Primary + 1d HTF — KAMA Adaptive Trend + Connors RSI + 1d HMA Filter

Hypothesis: After 8 failed experiments, the pattern is clear:
1. Overfiltering causes 0 trades (Sharpe=0.000 in #008)
2. Complex regime switching (CHOP+ADX+multiple filters) reduces trade count too much
3. Fisher Transform failed in #005 (Sharpe=-0.126)
4. Simpler is better: 1d trend bias + 4h adaptive entry + CRSI timing

This strategy uses PROVEN components from quantitative literature:
- Kaufman Adaptive Moving Average (KAMA): adjusts speed based on market efficiency
  - Fast in trends, slow in chop — perfect for 2022 whipsaw + 2025 bear
- Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
  - Literature shows 75% win rate for mean-reversion entries
  - Long: CRSI < 20, Short: CRSI > 80 (wider than failed attempts)
- 1d HMA(50): Major trend bias only — much slower than 4h for stability
- ATR(14) stoploss: 2.5x trailing stop

Why this should beat previous failures:
- KAMA adapts to volatility — doesn't whipsaw in 2022 crash like EMA
- CRSI < 20 / > 80 is WIDER than Fisher thresholds — ensures trades happen
- 1d HMA is SLOW trend filter — prevents counter-trend trades without overfiltering
- Single regime logic (no ADX hysteresis complexity that killed #005)
- Position size 0.30 (discrete) — conservative for 4h timeframe

Target: 25-45 trades/year on 4h (per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_hma1d_v1"
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
    
    KAMA adapts to market efficiency:
    - High efficiency (trending) → moves fast toward price
    - Low efficiency (choppy) → moves slowly, filters noise
    
    Perfect for 2022 whipsaw + 2025 bear market.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER): |net change| / sum of absolute changes
    net_change = close_s.diff(er_period).abs()
    sum_changes = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close_s.iloc[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=50):
    """Calculate Hull Moving Average (HMA) for trend bias."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Percentile rank of price change over 100 periods
    
    Entry signals (literature):
    - Long: CRSI < 20 (oversold)
    - Short: CRSI > 80 (overbought)
    
    This is WIDER than typical RSI to ensure trades happen.
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank of price change
    price_change = close_s.pct_change()
    percent_rank = price_change.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) if len(x) >= rank_period else np.nan
    )
    rsi_rank = percent_rank * 100.0
    
    # Combine components
    crsi = (rsi_short + rsi_streak + rsi_rank) / 3.0
    
    return crsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend bias (SLOW filter)
    hma_1d = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(crsi[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (SLOW) ===
        # Price above 1d HMA = bullish bias, below = bearish bias
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # 1d HMA slope (5-bar lookback for stability)
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # === 4H KAMA TREND ===
        # KAMA slope indicates short-term momentum
        kama_slope_bull = kama_4h[i] > kama_4h[i-3] if i >= 3 else False
        kama_slope_bear = kama_4h[i] < kama_4h[i-3] if i >= 3 else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # === CONNORS RSI SIGNALS (WIDER thresholds for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Long entry
        crsi_overbought = crsi[i] > 80.0  # Short entry
        
        # CRSI exiting extremes (for confirmation)
        crsi_rising_from_low = crsi[i] > crsi[i-1] and crsi[i-1] < 25.0 if i >= 1 else False
        crsi_falling_from_high = crsi[i] < crsi[i-1] and crsi[i-1] > 75.0 if i >= 1 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Must have 1d bullish bias (not strongly bearish)
        # AND 4h KAMA showing momentum
        # AND CRSI oversold
        if price_above_hma_1d or (not hma_1d_slope_bear):
            if kama_slope_bull or price_above_kama:
                if crsi_oversold or crsi_rising_from_low:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Must have 1d bearish bias (not strongly bullish)
        # AND 4h KAMA showing downward momentum
        # AND CRSI overbought
        if price_below_hma_1d or (not hma_1d_slope_bull):
            if kama_slope_bear or price_below_kama:
                if crsi_overbought or crsi_falling_from_high:
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
        # Exit long if 1d trend turns bearish
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if 1d trend turns bullish
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals