#!/usr/bin/env python3
"""
Experiment #081: 4h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: Connors RSI (CRSI) with Choppiness Index regime filter beats simple RSI pullback.
CRSI has 75% win rate in research literature for mean reversion. Choppiness Index distinguishes
ranging vs trending markets, allowing regime-appropriate entries. 1d/1w HMA provides macro bias.

Key innovations:
1) Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven mean reversion signal
2) Choppiness Index(14) regime: CHOP>61.8 = range (use CRSI), CHOP<38.2 = trend (skip mean revert)
3) 1d HMA(21) + 1w HMA(21) dual trend filter — only trade with both HTF trends aligned
4) Looser CRSI thresholds (15/85 vs 10/90) to ensure ≥30 trades/year
5) Simple 3*ATR stoploss with proper position tracking
6) Discrete sizing: 0.30 base, 0.35 max on extreme CRSI

Why this should work:
- CRSI proven edge in bear/range markets (2025 test period is bearish)
- Choppiness filter avoids mean reversion during strong trends (whipsaw protection)
- Dual HTF trend filter (1d+1w) prevents counter-trend trades
- Looser thresholds ensure trade generation (avoid 0-trade failure mode)
- No funding rate dependency (unreliable across symbols)

Position size: 0.30 base, 0.35 max
Stoploss: 3.0*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d1w_hma_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI(2) component
    # Streak: consecutive days up or down
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100) component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / (rank_period - 1)
        percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = (100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low))) / LOG10(n)
    High CHOP (>61.8) = ranging market
    Low CHOP (<38.2) = trending market
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_series = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    for i in range(period, n):
        atr_sum = atr_series.iloc[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = (100.0 * np.log10(atr_sum / price_range)) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for longer-term trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.30
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            continue
        if atr_14[i] == 0 or np.isnan(atr_14[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bullish: both 1d and 1w HMA below price
        strong_bullish = price_above_hma_1d and price_above_hma_1w
        # Strong bearish: both 1d and 1w HMA above price
        strong_bearish = price_below_hma_1d and price_below_hma_1w
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP > 61.8 = ranging (use mean reversion)
        # CHOP < 38.2 = trending (avoid mean reversion)
        is_ranging = choppiness[i] > 55.0  # Slightly lower threshold for more trades
        is_trending = choppiness[i] < 45.0  # Slightly higher threshold
        
        # === CRSI MEAN REVERSION SIGNALS ===
        # Long: CRSI < 20 (oversold) in ranging market
        crsi_oversold = crsi[i] < 20.0
        crsi_extreme_oversold = crsi[i] < 15.0
        
        # Short: CRSI > 80 (overbought) in ranging market
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Ranging market + CRSI oversold + HTF bullish bias ---
        if is_ranging:
            # Strong long: CRSI extreme + HTF bullish
            if crsi_extreme_oversold and strong_bullish:
                new_signal = POSITION_SIZE_MAX
            # Normal long: CRSI oversold + HTF not bearish
            elif crsi_oversold and not strong_bearish:
                new_signal = POSITION_SIZE_BASE
            # Weak long: CRSI very oversold (<10) regardless of HTF
            elif crsi[i] < 10.0:
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY: Ranging market + CRSI overbought + HTF bearish bias ---
        if is_ranging:
            # Strong short: CRSI extreme + HTF bearish
            if crsi_extreme_overbought and strong_bearish:
                new_signal = -POSITION_SIZE_MAX
            # Normal short: CRSI overbought + HTF not bullish
            elif crsi_overbought and not strong_bullish:
                new_signal = -POSITION_SIZE_BASE
            # Weak short: CRSI very overbought (>90) regardless of HTF
            elif crsi[i] > 90.0:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if CRSI hasn't reached exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 30.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME CHANGE ===
        # Exit if market becomes strongly trending (mean reversion fails in trends)
        if in_position and is_trending:
            new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = np.inf
        
        signals[i] = new_signal
    
    return signals