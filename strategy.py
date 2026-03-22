#!/usr/bin/env python3
"""
Experiment #542: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + Connors RSI

Hypothesis: After 480+ failed strategies, the key insight is REGIME ADAPTATION.
Simple trend-following fails in bear/range markets (2022 crash, 2025 bear).
Pure mean-reversion fails in strong trends. We need BOTH with regime detection.

Key insights from research:
- Choppiness Index (CHOP) is the BEST regime filter for crypto
- CHOP > 61.8 = range market → use Connors RSI mean reversion
- CHOP < 38.2 = trending market → use HMA trend following
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- 1w HMA for major trend bias (prevents counter-trend mean reversion)
- 1d HMA for intermediate trend confirmation

This strategy uses:
1. 1w HMA(21) for major trend bias (HTF filter - call ONCE before loop)
2. 1d HMA(21) for intermediate trend (HTF filter - call ONCE before loop)
3. 12h Choppiness Index(14) for regime detection
4. 12h Connors RSI for mean reversion entries in chop
5. 12h HMA(16/48) crossover for trend entries in trend regime
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete position sizing (0.30) to minimize fee churn

Why this might work:
- Dual regime adapts to bull/bear/range markets automatically
- Connors RSI has 75% win rate in range markets (research-backed)
- 1w/1d HTF filters prevent dangerous counter-trend trades
- 12h TF targets 20-50 trades/year (optimal fee/trade ratio)
- Simpler than volspike/Fisher combos that failed 480+ times

Position sizing: 0.30 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_connors_hma_1d1w_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Range/Choppy market (mean reversion)
    - CHOP < 38.2 = Trending market (trend following)
    - 38.2 < CHOP < 61.8 = Transition zone
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # Neutral if no range
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Consecutive up/down days
    PercentRank: Where current close ranks vs last 100 closes
    
    Entry signals:
    - Long: CRSI < 10 (oversold)
    - Short: CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_window > 0)
        down_streaks = np.sum(streak_window < 0)
        if up_streaks + down_streaks > 0:
            streak_rsi[i] = 100.0 * up_streaks / (up_streaks + down_streaks)
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: Percent Rank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Combine components
    for i in range(rank_period, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    # Fill early values
    crsi[:rank_period] = 50.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF HMA for intermediate trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    HALF_POSITION = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track HMA crossover state
    prev_hma_16 = np.zeros(n)
    prev_hma_16[1:] = hma_12h_16[:-1]
    prev_hma_48 = np.zeros(n)
    prev_hma_48[1:] = hma_12h_48[:-1]
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 55.0  # Range market (slightly lower threshold for more trades)
        trending_regime = chop_14[i] < 45.0  # Trending market
        # 45-55 = transition zone (use trend logic as default)
        
        # === 1W MAJOR TREND BIAS ===
        bull_bias_1w = close[i] > hma_1w_aligned[i]
        bear_bias_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        bull_trend_1d = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_trend_1d = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        price_above_1d = close[i] > hma_1d_21_aligned[i]
        price_below_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA CROSSOVER (trend following) ===
        hma_bull_cross = (hma_12h_16[i] > hma_12h_48[i]) and (prev_hma_16[i] <= prev_hma_48[i])
        hma_bear_cross = (hma_12h_16[i] < hma_12h_48[i]) and (prev_hma_16[i] >= prev_hma_48[i])
        hma_bull = hma_12h_16[i] > hma_12h_48[i]
        hma_bear = hma_12h_16[i] < hma_12h_48[i]
        
        # === CONNORS RSI (mean reversion) ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        crsi_recovering_long = crsi[i] < 30.0 and crsi[i] > crsi[i-1] if i > 0 else False
        crsi_recovering_short = crsi[i] > 70.0 and crsi[i] < crsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # --- TREND REGIME (CHOP < 45) ---
        if trending_regime:
            # LONG: Trend + HTF confirmation + HMA bull
            if bull_bias_1w and bull_trend_1d and hma_bull:
                new_signal = POSITION_SIZE
            # Entry on HMA crossover with HTF confirmation
            elif bull_bias_1w and hma_bull_cross:
                new_signal = POSITION_SIZE
            # Pullback entry in uptrend
            elif bull_bias_1w and bull_trend_1d and price_below_1d and hma_bull:
                new_signal = HALF_POSITION
            
            # SHORT: Trend + HTF confirmation + HMA bear
            if new_signal == 0.0:
                if bear_bias_1w and bear_trend_1d and hma_bear:
                    new_signal = -POSITION_SIZE
                # Entry on HMA crossover with HTF confirmation
                elif bear_bias_1w and hma_bear_cross:
                    new_signal = -POSITION_SIZE
                # Pullback entry in downtrend
                elif bear_bias_1w and bear_trend_1d and price_above_1d and hma_bear:
                    new_signal = -HALF_POSITION
        
        # --- CHOPPY REGIME (CHOP > 55) ---
        elif choppy_regime:
            # LONG: Connors RSI oversold + 1w bull bias (don't short in bull macro)
            if bull_bias_1w and crsi_oversold:
                new_signal = POSITION_SIZE
            # LONG: Connors RSI recovering from oversold + 1w bull bias
            elif bull_bias_1w and crsi_recovering_long:
                new_signal = HALF_POSITION
            # LONG: Deep oversold even in bear (mean reversion bounce)
            elif crsi[i] < 10.0:
                new_signal = HALF_POSITION
            
            # SHORT: Connors RSI overbought + 1w bear bias (don't long in bear macro)
            if new_signal == 0.0:
                if bear_bias_1w and crsi_overbought:
                    new_signal = -POSITION_SIZE
                # SHORT: Connors RSI recovering from overbought + 1w bear bias
                elif bear_bias_1w and crsi_recovering_short:
                    new_signal = -HALF_POSITION
                # SHORT: Deep overbought even in bull (mean reversion drop)
                elif crsi[i] > 90.0:
                    new_signal = -HALF_POSITION
        
        # --- TRANSITION ZONE (45 <= CHOP <= 55) ---
        else:
            # Use trend logic as default but with reduced size
            if bull_bias_1w and hma_bull:
                new_signal = HALF_POSITION
            elif bear_bias_1w and hma_bear:
                new_signal = -HALF_POSITION
        
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
        
        # === EXIT CONDITIONS (regime flip or trend reversal) ===
        # Exit long on major regime flip
        if in_position and position_side > 0:
            if bear_bias_1w and bear_trend_1d:
                new_signal = 0.0
            # Exit on HMA bear cross
            elif hma_bear_cross:
                new_signal = 0.0
            # Exit mean reversion on CRSI recovery
            elif choppy_regime and crsi[i] > 60.0:
                new_signal = 0.0
        
        # Exit short on major regime flip
        if in_position and position_side < 0:
            if bull_bias_1w and bull_trend_1d:
                new_signal = 0.0
            # Exit on HMA bull cross
            elif hma_bull_cross:
                new_signal = 0.0
            # Exit mean reversion on CRSI recovery
            elif choppy_regime and crsi[i] < 40.0:
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