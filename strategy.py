#!/usr/bin/env python3
"""
Experiment #553: 1d Primary + 1w HTF — Dual Regime (Chop/Trend) with Connors RSI

Hypothesis: Based on research showing Connors RSI works well in range markets (75% win rate)
and Choppiness Index effectively distinguishes regime. Combined with 1w major trend filter
and Donchian breakout for trending regimes, this should outperform simple HMA strategies.

Key innovations vs failed attempts:
1. DUAL REGIME: Mean reversion when CHOP>61.8, trend follow when CHOP<38.2
2. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven 75% win rate
3. 1w HMA bias: Only take signals aligned with weekly trend direction
4. Donchian(20) breakout: For trend regime entries
5. Asymmetric sizing: 0.30 with trend, 0.20 counter-trend (rare)

Why this might beat Sharpe=0.435:
- 1d timeframe targets 20-50 trades/year (optimal per rules)
- Regime switching adapts to market conditions (range vs trend)
- Connors RSI catches reversals better than standard RSI(14)
- 1w HTF prevents major counter-trend losses
- Simpler than failed 4h/12h complex strategies

Position sizing: 0.20-0.30 discrete (Rule 4, max 0.40)
Stoploss: 2.5 * ATR(14) trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_connors_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak: Consecutive days of gains/losses (positive for gains, negative for losses)
    PercentRank: Percentile rank of daily return over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = max(0, streak[i-1]) + 1
        elif delta.iloc[i] < 0:
            streak[i] = min(0, streak[i-1]) - 1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank of daily returns
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if pd.notna(current):
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100
        else:
            percent_rank.iloc[i] = 50.0
    
    # Fill early values
    percent_rank.iloc[:rank_period] = 50.0
    
    # Calculate CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    choppiness = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # 1d HMA for intermediate trend
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    # Donchian channels for breakout detection
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.30      # Full size with trend
    SIZE_COUNTER = 0.20    # Reduced size counter-trend (rare)
    SIZE_CHOP = 0.25       # Medium size in chop regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        hma_1w_slope_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        hma_1w_slope_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        bull_regime_1d = close[i] > hma_1d_21[i]
        bear_regime_1d = close[i] < hma_1d_21[i]
        
        hma_1d_slope_bull = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_slope_bear = hma_1d_21[i] < hma_1d_50[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (breakout)
        # 38.2 <= CHOP <= 61.8 = transition (no trades)
        chop_regime = choppiness[i] > 61.8
        trend_regime = choppiness[i] < 38.2
        transition_regime = not chop_regime and not trend_regime
        
        # === VOLATILITY FILTER ===
        vol_ratio = atr_14[i] / (atr_50[i] + 1e-10)
        vol_ok = vol_ratio > 0.6  # At least 60% of long-term avg
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        if not transition_regime and vol_ok:
            # --- CHOPPY REGIME: Mean Reversion with Connors RSI ---
            if chop_regime:
                # Long: CRSI < 10 (extreme oversold) + 1w bull bias preferred
                if crsi[i] < 12.0:
                    if bull_regime_1w:
                        new_signal = SIZE_TREND
                    elif bear_regime_1w:
                        new_signal = SIZE_COUNTER  # Counter-trend, smaller size
                
                # Short: CRSI > 90 (extreme overbought) + 1w bear bias preferred
                elif crsi[i] > 88.0:
                    if bear_regime_1w:
                        new_signal = -SIZE_TREND
                    elif bull_regime_1w:
                        new_signal = -SIZE_COUNTER
            
            # --- TRENDING REGIME: Donchian Breakout ---
            elif trend_regime:
                # Long breakout: Price breaks Donchian upper + 1w/1d bull
                if close[i] > donchian_upper[i] * 0.998:  # Near breakout
                    if bull_regime_1w and bull_regime_1d:
                        new_signal = SIZE_TREND
                    elif bull_regime_1w:
                        new_signal = SIZE_TREND * 0.8
                
                # Short breakout: Price breaks Donchian lower + 1w/1d bear
                elif close[i] < donchian_lower[i] * 1.002:  # Near breakout
                    if bear_regime_1w and bear_regime_1d:
                        new_signal = -SIZE_TREND
                    elif bear_regime_1w:
                        new_signal = -SIZE_TREND * 0.8
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        # Exit long on 1w regime flip to strong bear
        if in_position and position_side > 0:
            if bear_regime_1w and hma_1w_slope_bear:
                new_signal = 0.0
            # Exit on CRSI extreme overbought in chop regime
            elif chop_regime and crsi[i] > 85.0:
                new_signal = 0.0
        
        # Exit short on 1w regime flip to strong bull
        if in_position and position_side < 0:
            if bull_regime_1w and hma_1w_slope_bull:
                new_signal = 0.0
            # Exit on CRSI extreme oversold in chop regime
            elif chop_regime and crsi[i] < 15.0:
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