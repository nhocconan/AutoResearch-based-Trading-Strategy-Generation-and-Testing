#!/usr/bin/env python3
"""
Experiment #517: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI Pullback + ATR Trail

Hypothesis: After 448 failed strategies (mostly vol-spike/Fisher/Choppiness/CRSI combos),
try a SIMPLER approach that generates MORE trades while maintaining edge:

1. KAMA (Kaufman Adaptive Moving Average): Adapts to market efficiency ratio.
   Reduces whipsaws in choppy markets (2022-2025 range/bear) while catching trends.
   More robust than EMA/HMA for BTC/ETH which fail simple trend strategies.

2. RSI(7) PULLBACK entries (not extremes): RSI(7)<40 in uptrend, RSI(7)>60 in downtrend.
   Looser than RSI<30/>70 to generate MORE trades (critical: need >=30/symbol on train).
   Many failed strategies had 0 trades due to overly strict entry conditions.

3. 1w HMA(21) as major trend filter: Only trade in direction of weekly trend.
   Prevents counter-trend trades that got crushed in 2022 bear market.

4. ATR(14) 2.5x trailing stop: Protects capital in crash scenarios.
   Signal → 0 when price moves 2.5*ATR against position.

Why this might beat current best (Sharpe=0.435):
- KAMA is DIFFERENT from HMA/EMA (adaptive, not fixed smoothing)
- Looser RSI thresholds = more trades (addresses #1 failure mode: 0 trades)
- 1d TF targets 20-50 trades/year (lower fee drag, matches proven patterns)
- Simpler logic = fewer conflicting filters that cancel each other out
- 1w HTF filter prevents bear market disasters (2022 -77% crash)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi7_pullback_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio (ER).
    High ER (trending) = fast smoothing. Low ER (choppy) = slow smoothing.
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |Price Change| / Sum of |Individual Changes|
    price_change = np.abs(close_s.diff(period))
    sum_changes = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    
    # Avoid division by zero
    sum_changes = sum_changes.replace(0, 1e-10)
    er = price_change / sum_changes
    er = er.fillna(0)
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # KAMA smoothing constant = ER * (fast_sc - slow_sc) + slow_sc
    sc = (er.values * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA iteratively
    kama = np.zeros(n)
    kama[0] = close_s.iloc[0]
    
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) for HTF trend."""
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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period (use 7 for faster signals)."""
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
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    # Count consecutive up/down days
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            if i > 0 and delta.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta.iloc[i] < 0:
            if i > 0 and delta.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert to RSI-like scale (0-100)
    streak_s = pd.Series(streak)
    streak_rsi = calculate_rsi(np.abs(streak), period)
    
    # Adjust for direction
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component for Connors RSI.
    Measures where current return ranks vs past N days.
    """
    close_s = pd.Series(close)
    returns = close_s.pct_change()
    
    percent_rank = np.zeros(len(close))
    
    for i in range(period, len(close)):
        if np.isnan(returns.iloc[i]):
            percent_rank[i] = 50.0
            continue
        
        current_return = returns.iloc[i]
        past_returns = returns.iloc[i-period+1:i]
        
        # Count how many past returns are less than current
        rank = np.sum(past_returns < current_return)
        percent_rank[i] = (rank / period) * 100.0
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    rsi_fast = calculate_rsi(close, rsi_period)
    # streak_rsi = calculate_rsi_streak(close, streak_period)
    # percent_rank = calculate_percent_rank(close, pr_period)
    
    # Simplified CRSI (just fast RSI for speed + trade frequency)
    crsi = rsi_fast
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    kama_10 = calculate_kama(close, period=10)
    kama_20 = calculate_kama(close, period=20)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_10[i]) or np.isnan(kama_20[i]):
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(rsi_7[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D KAMA TREND (entry timing) ===
        kama_bull = kama_10[i] > kama_20[i]
        kama_bear = kama_10[i] < kama_20[i]
        
        # KAMA slope (momentum)
        kama_slope_up = kama_10[i] > kama_10[i-5] if i >= 5 else False
        kama_slope_down = kama_10[i] < kama_10[i-5] if i >= 5 else False
        
        # === RSI PULLBACK SIGNALS (looser thresholds for more trades) ===
        rsi_pullback_long = rsi_7[i] < 40.0  # Pullback in uptrend
        rsi_pullback_short = rsi_7[i] > 60.0  # Bounce in downtrend
        rsi_extreme_low = rsi_7[i] < 30.0  # Oversold
        rsi_extreme_high = rsi_7[i] > 70.0  # Overbought
        
        # === ENTRY LOGIC — KAMA TREND + RSI PULLBACK ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions to ensure trade frequency)
        # Condition 1: Weekly bull + KAMA bull + RSI pullback (primary setup)
        if bull_regime and kama_bull and rsi_pullback_long:
            new_signal = LONG_SIZE
        # Condition 2: Weekly bull + RSI extreme (strong reversal signal)
        elif bull_regime and rsi_extreme_low:
            new_signal = LONG_SIZE
        # Condition 3: KAMA crossover bull + RSI pullback (momentum entry)
        elif kama_bull and kama_slope_up and rsi_pullback_long:
            new_signal = LONG_SIZE * 0.8
        # Condition 4: Price > KAMA + RSI not overbought (trend continuation)
        elif close[i] > kama_10[i] and rsi_7[i] < 65.0 and bull_regime:
            new_signal = LONG_SIZE * 0.6
        # Condition 5: RSI extreme alone (capitulation long)
        elif rsi_extreme_low and rsi_7[i] < 25.0:
            new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (mirror logic for bear market)
        if new_signal == 0.0:
            # Condition 1: Weekly bear + KAMA bear + RSI bounce (primary setup)
            if bear_regime and kama_bear and rsi_pullback_short:
                new_signal = -SHORT_SIZE
            # Condition 2: Weekly bear + RSI extreme (strong reversal signal)
            elif bear_regime and rsi_extreme_high:
                new_signal = -SHORT_SIZE
            # Condition 3: KAMA crossover bear + RSI bounce (momentum entry)
            elif kama_bear and kama_slope_down and rsi_pullback_short:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 4: Price < KAMA + RSI not oversold (trend continuation)
            elif close[i] < kama_10[i] and rsi_7[i] > 35.0 and bear_regime:
                new_signal = -SHORT_SIZE * 0.6
            # Condition 5: RSI extreme alone (FOMO short)
            elif rsi_extreme_high and rsi_7[i] > 75.0:
                new_signal = -SHORT_SIZE * 0.7
        
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
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on RSI overbought or regime flip
        if in_position and position_side > 0:
            if rsi_extreme_high:
                new_signal = 0.0
            # Exit if weekly regime flips bearish
            if bear_regime and kama_bear:
                new_signal = 0.0
        
        # Exit short on RSI oversold or regime flip
        if in_position and position_side < 0:
            if rsi_extreme_low:
                new_signal = 0.0
            # Exit if weekly regime flips bullish
            if bull_regime and kama_bull:
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