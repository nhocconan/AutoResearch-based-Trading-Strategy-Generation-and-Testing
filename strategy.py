#!/usr/bin/env python3
"""
Experiment #647: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Building on proven 1d patterns (Choppiness+Connors RSI showed ETH Sharpe +0.923,
HMA+RSI+Donchian showed SOL Sharpe +0.879), this strategy uses regime-adaptive logic:
- CHOPPY regime (Choppiness > 61.8): Mean reversion via Connors RSI extremes
- TREND regime (Choppiness < 38.2): Trend follow via HMA + Donchian breakout

Key insights from 572 failed strategies:
1. 1d timeframe with 1w HTF works well for major trend filter (not too fast, not too slow)
2. Choppiness Index is the BEST regime filter for bear/range markets (2022-2024)
3. Connors RSI catches reversals better than standard RSI (75% win rate documented)
4. Dual regime approach adapts to market conditions instead of forcing one style
5. Conservative sizing (0.30) + ATR stop controls drawdown through 2022 crash
6. Simpler entry logic = more trades (avoid 0-trade failure mode)

Why this might beat Sharpe=0.520:
- 1w HMA slope keeps us on right side of multi-month moves
- Choppiness regime detection switches between mean-revert and trend-follow
- Connors RSI (RSI3 + RSI_Streak + PercentRank) / 3 is proven reversal indicator
- Donchian(20) breakout confirms trend strength before trend-follow entry
- 2.5*ATR trailing stop limits losses on reversals
- Target 25-40 trades/year on 1d (per Rule 10 for daily timeframe)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-40 trades/year on 1d
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_1w_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    Values > 61.8 = choppy/range, < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    
    # Avoid division by zero
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI applied to consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on price
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak (consecutive up/down days)
    returns = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: PercentRank of returns over last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period+1:i+1].values
        today_return = returns.iloc[i]
        rank = np.sum(window_returns < today_return)
        percent_rank[i] = rank / rank_period * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bands.
    Upper = highest high over period
    Lower = lowest low over period
    """
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
    
    # Calculate 1w HMA for primary trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=9)
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_choppy = chop_14[i] > 55.0  # Range/mean-revert regime
        regime_trend = chop_14[i] < 45.0   # Trend-follow regime
        # Neutral zone 45-55: no new entries, hold existing
        
        # === 1W TREND BIAS (HMA slope over 2 bars) ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # Price relative to 1w HMA
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA FAST/SLOW CROSSOVER ===
        hma_cross_bull = hma_1d_fast[i] > hma_1d[i]
        hma_cross_bear = hma_1d_fast[i] < hma_1d[i]
        
        # === 1D HMA SLOPE (2 bars) ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-2] if i >= 2 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === CONNORS RSI EXTREMES (Mean Reversion) ===
        crsi_oversold = crsi[i] < 15.0   # Strong buy signal
        crsi_overbought = crsi[i] > 85.0  # Strong sell signal
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- MEAN REVERSION ENTRY (Choppy Regime) ---
        # Connors RSI extremes work best in range-bound markets
        if regime_choppy:
            # Long: CRSI oversold + price above 1w HMA (bullish bias)
            if crsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE
            # Short: CRSI overbought + price below 1w HMA (bearish bias)
            elif crsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE
        
        # --- TREND FOLLOW ENTRY (Trending Regime) ---
        # HMA crossover + Donchian breakout in trend direction
        elif regime_trend:
            # Long: 1w bull trend + 1d HMA cross up + Donchian breakout
            if hma_1w_slope_bull and price_above_hma_1w:
                if hma_cross_bull and hma_1d_slope_bull:
                    if donchian_breakout_up:
                        new_signal = POSITION_SIZE
            # Short: 1w bear trend + 1d HMA cross down + Donchian breakout
            elif hma_1w_slope_bear and price_below_hma_1w:
                if hma_cross_bear and hma_1d_slope_bear:
                    if donchian_breakout_down:
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
            if hma_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1w_slope_bull and price_above_hma_1w:
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