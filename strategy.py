#!/usr/bin/env python3
"""
Experiment #612: 12h Primary + 1d/1w HTF — HMA Trend + Connors RSI + Choppiness Regime + Donchian

Hypothesis: Building on #606 (12h KAMA+CRSI+CHOP, Sharpe=0.179) and current best 
mtf_1d_chop_crsi_regime_1w_v1 (Sharpe=0.520), this strategy combines HMA trend following 
with Connors RSI for precise entries, regime-switching via Choppiness Index, and Donchian 
breakout confirmation for higher-probability trades.

Key insights from 541 failed strategies:
1. #607 (1d KAMA+CHOP+RSI) failed with Sharpe=-0.627 — KAMA too slow on 1d
2. HMA is more responsive than KAMA for trend detection (less lag)
3. Connors RSI (CRSI) outperforms regular RSI for mean reversion (75% win rate documented)
4. Donchian breakout adds momentum confirmation, reduces false signals
5. 12h timeframe targets 20-50 trades/year — need balanced entry filters
6. Asymmetric crypto behavior: longs need deeper oversold, shorts need moderate overbought

Why this might beat Sharpe=0.520:
- HMA(21) on 12h + 1d HTF trend filter reduces lag vs KAMA
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
- Choppiness regime switch: trend-follow when CHOP<45, mean-revert when CHOP>55
- Donchian(20) breakout confirmation ensures momentum alignment
- 1w HTF HMA slope prevents counter-trend trades during major moves
- Conservative size (0.30) with 2.5*ATR trailing stop controls drawdown
- Entry filters balanced to ensure 20-50 trades/year (not too strict like #605/#608)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 12h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_chop_donchian_1d1w_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA with less lag.
    """
    n = period
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half_n)
    wma_full = wma(close_s, n)
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback period
    
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streaks
    # Streak = consecutive up (positive) or down (negative) days
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
    # Positive streak = bullish, negative = bearish
    streak_series = pd.Series(streak)
    streak_gain = streak_series.where(streak_series > 0, 0.0)
    streak_loss = -streak_series.where(streak_series < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # Component 3: PercentRank of price change
    price_change = close_s.pct_change() * 100
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = price_change.iloc[i-rank_period:i]
        current = price_change.iloc[i]
        rank = (window < current).sum() / len(window) * 100
        percent_rank.iloc[i] = rank
    
    percent_rank = percent_rank.fillna(50.0).values
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (HMA slope over 3 bars) ===
        kama_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-3] if i >= 3 else False
        kama_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1w HMA
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA SLOPE (2 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-2] if i >= 2 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H HMA CROSSOVER ===
        hma_cross_bull = hma_12h_fast[i] > hma_12h[i]
        hma_cross_bear = hma_12h_fast[i] < hma_12h[i]
        
        # === 12H HMA SLOPE ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-2] if i >= 2 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-2] if i >= 2 else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0
        is_chop_regime = chop_14[i] > 55.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow HTF trend with 12h pullback entries ---
        if is_trend_regime:
            # LONG: 1w bull + 1d bull + 12h bull + CRSI pullback (30-50) + Donchian confirmation
            if kama_1w_slope_bull and hma_1d_slope_bull and hma_12h_slope_bull:
                if price_above_hma_1w and price_above_hma_1d:
                    if 25.0 <= crsi[i] <= 55.0:
                        if donchian_breakout_long or hma_cross_bull:
                            new_signal = POSITION_SIZE
            
            # SHORT: 1w bear + 1d bear + 12h bear + CRSI bounce (50-75) + Donchian confirmation
            elif kama_1w_slope_bear and hma_1d_slope_bear and hma_12h_slope_bear:
                if price_below_hma_1w and price_below_hma_1d:
                    if 50.0 <= crsi[i] <= 75.0:
                        if donchian_breakout_short or hma_cross_bear:
                            new_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at CRSI extremes ---
        elif is_chop_regime:
            # LONG: CRSI < 25 (deep oversold) + price below 12h HMA
            if crsi[i] < 25.0 and price_below_hma_1d:
                new_signal = POSITION_SIZE
            
            # SHORT: CRSI > 75 (overbought) + price above 12h HMA
            elif crsi[i] > 75.0 and price_above_hma_1d:
                new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME (45-55): Wait for strong signals ---
        else:
            # Only enter on strong Donchian breakout with HTF alignment
            if kama_1w_slope_bull and hma_1d_slope_bull and donchian_breakout_long:
                if crsi[i] < 50.0:
                    new_signal = POSITION_SIZE
            elif kama_1w_slope_bear and hma_1d_slope_bear and donchian_breakout_short:
                if crsi[i] > 50.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if kama_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_1w_slope_bull and price_above_hma_1w:
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