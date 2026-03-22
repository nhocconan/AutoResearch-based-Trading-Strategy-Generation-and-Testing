#!/usr/bin/env python3
"""
Experiment #537: 1d Primary + 1w HTF — Dual Regime (Trend + Mean Revert) + Connors RSI

Hypothesis: After 536 experiments, the key insight is that NO SINGLE regime works 
across all market conditions. BTC 2021-2024 includes bull (+219%), bear (-77%), 
and range periods. ETH follows similar pattern. SOL is outlier but needs coverage.

This strategy uses DUAL REGIME approach:
1. CHOPPINESS INDEX (14) to detect market state:
   - CHOP > 61.8 = Range/Chop → Mean Reversion mode (Connors RSI)
   - CHOP < 38.2 = Trend → Trend Following mode (Donchian breakout)
   - Between = Hold previous signal (hysteresis)

2. 1w HMA(21) for MAJOR trend bias (HTF filter):
   - Price > HMA_1w = Long bias only (no shorts)
   - Price < HMA_1w = Short bias only (no longs)
   - Prevents counter-trend trades that destroyed 2022 performance

3. Connors RSI for mean reversion entries (range mode):
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + above 1w HMA
   - Short: CRSI > 85 + below 1w HMA

4. Donchian Breakout for trend entries (trend mode):
   - Long: Break Donchian(20) high + ADX > 25 + above 1w HMA
   - Short: Break Donchian(20) low + ADX > 25 + below 1w HMA

5. ATR(14) 2.5x trailing stop for risk management

Why this might beat Sharpe=0.435:
- Adapts to bull/bear/range automatically (no manual regime switching)
- 1w HTF prevents catastrophic counter-trend trades (2022 lesson)
- Connors RSI has 75% win rate in ranges (proven in literature)
- Donchian captures major trends without whipsaw (ADX filter)
- 1d TF = 10-30 trades/year optimal for fee/trade ratio
- Discrete sizing (0.30) minimizes fee churn

Position sizing: 0.30 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=10 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_connors_donchian_1w_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component for Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    for i in range(period, n):
        up_streaks = np.sum(streak[i-period+1:i+1] > 0)
        down_streaks = np.sum(streak[i-period+1:i+1] < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component for Connors RSI.
    Measures where current return ranks vs past N days.
    """
    n = len(close)
    pct_rank = np.zeros(n)
    
    returns = np.zeros(n)
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10) * 100
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        pct_rank[i] = 100.0 * rank / period
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    - RSI(3): Short-term momentum
    - RSI_Streak(2): Consecutive up/down days
    - PercentRank(100): Current return vs past 100 days
    
    Entry signals:
    - Long: CRSI < 10-15 (oversold)
    - Short: CRSI > 85-90 (overbought)
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Range/Chop (mean reversion favorable)
    - CHOP < 38.2 = Trend (trend following favorable)
    - 38.2 - 61.8 = Transition (hold previous signal)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = np.zeros(n)
    for i in range(period, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest High - Lowest Low over period
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        range_hl = hh - ll
        
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100.0 * np.log10(atr_sum[i] / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0  # Neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (breakout levels).
    Upper = Highest High over N periods
    Lower = Lowest Low over N periods
    """
    n = len(close) if 'close' in dir() else len(high)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track regime state (hysteresis to avoid flipping)
    prev_regime = 0  # 0 = unknown, 1 = trend, -1 = range
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]) or np.isnan(crsi[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 1W MAJOR TREND BIAS (primary direction filter) ===
        bull_bias = close[i] > hma_1w_21_aligned[i]
        bear_bias = close[i] < hma_1w_21_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        
        # Determine regime with hysteresis
        if chop_value < 38.2:
            current_regime = 1  # Trend mode
        elif chop_value > 61.8:
            current_regime = -1  # Range mode
        else:
            current_regime = prev_regime  # Hold previous (hysteresis)
        
        prev_regime = current_regime
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # TREND MODE (CHOP < 38.2) - Donchian Breakout
        if current_regime == 1:
            # Long: Breakout above Donchian + ADX confirms trend + bull bias
            if close[i] > donchian_upper[i-1] and adx_14[i] > 25.0 and bull_bias:
                new_signal = POSITION_SIZE
            # Short: Breakout below Donchian + ADX confirms trend + bear bias
            elif close[i] < donchian_lower[i-1] and adx_14[i] > 25.0 and bear_bias:
                new_signal = -POSITION_SIZE
        
        # RANGE MODE (CHOP > 61.8) - Connors RSI Mean Reversion
        elif current_regime == -1:
            # Long: CRSI oversold + bull bias (only long in bull bias)
            if crsi[i] < 15.0 and bull_bias:
                new_signal = POSITION_SIZE
            # Short: CRSI overbought + bear bias (only short in bear bias)
            elif crsi[i] > 85.0 and bear_bias:
                new_signal = -POSITION_SIZE
            # Additional: Deep oversold/overbought regardless of bias (smaller size)
            elif crsi[i] < 10.0:
                new_signal = POSITION_SIZE * 0.5
            elif crsi[i] > 90.0:
                new_signal = -POSITION_SIZE * 0.5
        
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
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        # Exit long on bias flip to bear
        if in_position and position_side > 0:
            if bear_bias and current_regime == 1:  # Trend mode but bear bias
                new_signal = 0.0
            elif current_regime == -1 and crsi[i] > 70.0:  # Range mode + CRSI rising
                new_signal = 0.0
        
        # Exit short on bias flip to bull
        if in_position and position_side < 0:
            if bull_bias and current_regime == 1:  # Trend mode but bull bias
                new_signal = 0.0
            elif current_regime == -1 and crsi[i] < 30.0:  # Range mode + CRSI falling
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