#!/usr/bin/env python3
"""
Experiment #617: 1d Primary + 4h HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Building on current best mtf_1d_chop_crsi_regime_1w_v1 (Sharpe=0.520), this strategy
simplifies the entry logic while keeping the proven regime-switching framework. Key changes:

1. Use 4h HTF instead of 1w — faster trend signals, more trades (addressing 0-trade failures)
2. Connors RSI (CRSI) instead of standard RSI — proven 75% win rate for mean reversion
3. Simpler entry conditions — fewer AND conditions to ensure trades actually trigger
4. HMA instead of KAMA — faster response to trend changes, less lag
5. Asymmetric position sizing — larger in trend regime, smaller in chop regime

Why this might beat Sharpe=0.520:
- 4h trend filter is more responsive than 1w (captures intermediate trends)
- CRSI combines RSI(3) + RSI_Streak(2) + PercentRank(100) — better than single RSI
- Simpler logic = more trades (avoiding Sharpe=0.000 from no trades)
- HMA has less lag than KAMA for trend detection
- Proven components: CRSI (research-backed), CHOP (regime filter), HMA (trend)

Position sizing: 0.30 trend regime, 0.20 chop regime (discrete per Rule 4)
Target: 25-45 trades/year on 1d
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_4h_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI_Streak(2) — RSI of consecutive up/down days
    3. PercentRank(100) — percentile of today's return vs last 100 days
    
    CRSI < 10 = extremely oversold (long opportunity)
    CRSI > 90 = extremely overbought (short opportunity)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # Component 3: PercentRank of returns
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current_return = returns[i]
        if np.isnan(current_return):
            percent_rank[i] = 50.0
        else:
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                percent_rank[i] = 100.0 * np.sum(valid_window <= current_return) / len(valid_window)
            else:
                percent_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Much less lag than EMA/SMA while maintaining smoothness.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.30  # Larger size in trending regime
    SIZE_CHOP = 0.20   # Smaller size in choppy regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(crsi[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS (HMA slope) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-1] if i >= 1 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-1] if i >= 1 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D HMA SLOPE ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-1] if i >= 1 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-1] if i >= 1 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d[i]
        price_below_hma_1d = close[i] < hma_1d[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0
        is_chop_regime = chop_14[i] > 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        current_size = SIZE_TREND if is_trend_regime else SIZE_CHOP
        
        # --- TREND REGIME: Follow 4h trend with CRSI pullback entries ---
        if is_trend_regime:
            # LONG: 4h bull trend + CRSI oversold pullback (20-40)
            if hma_4h_slope_bull and price_above_hma_4h:
                if 15.0 <= crsi[i] <= 40.0:
                    new_signal = current_size
            
            # SHORT: 4h bear trend + CRSI overbought bounce (60-85)
            elif hma_4h_slope_bear and price_below_hma_4h:
                if 60.0 <= crsi[i] <= 85.0:
                    new_signal = -current_size
        
        # --- CHOP REGIME: Mean reversion at CRSI extremes ---
        elif is_chop_regime:
            # LONG: CRSI < 20 (extreme oversold)
            if crsi[i] < 20.0:
                new_signal = current_size
            
            # SHORT: CRSI > 80 (extreme overbought)
            elif crsi[i] > 80.0:
                new_signal = -current_size
        
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
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
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