#!/usr/bin/env python3
"""
Experiment #631: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Building on #624's +39.9% return (but poor Sharpe=-0.002), this strategy
simplifies entry logic while adding proven edges from research:
1. Connors RSI (CRSI) for mean reversion entries - 75% win rate in literature
2. Choppiness Index (CHOP) regime filter - avoid trend-following in choppy markets
3. 1d HMA for primary trend bias (simpler than 12h slope calculations)
4. Fewer confluence filters = more trades (target 30-50/year on 4h)
5. Tighter stoploss (2.0*ATR vs 2.5*ATR) and smaller size (0.25 vs 0.28)

Why this might beat Sharpe=0.520:
- #624 had good return but bad Sharpe = too much drawdown/volatility
- CHOP filter avoids whipsaw losses in ranging markets (major Sharpe killer)
- CRSI entries have higher win rate than simple RSI pullbacks
- Simpler logic = fewer false signals, cleaner exits
- 1d HTF trend filter keeps us on right side of major moves

Key differences from #624:
- Remove Donchian breakout (too many false breakouts in chop)
- Replace RSI(14) with Connors RSI (better for mean reversion)
- Add Choppiness Index regime detection (CHOP>61.8 = range, CHOP<38.2 = trend)
- Smaller position size (0.25) + tighter stops (2.0*ATR)
- Single 1d HMA filter instead of complex 12h slope calculations

Position sizing: 0.25 discrete (per Rule 4, max 0.40)
Target: 30-50 trades/year on 4h (per Rule 10)
Stoploss: 2.0*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_1d_v1"
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
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term momentum
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100.0 * (streak_abs[i] / (streak_abs[i] + 1e-10))
        else:
            streak_rsi[i] = 100.0 * (1.0 / (streak_abs[i] + 1e-10 + 1.0))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - where current price change ranks in lookback period
    pct_rank = np.zeros(n)
    returns = close_s.pct_change().values
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)  # avoid division by zero
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (mean reversion favored)
        is_trending = chop[i] < 45.0  # Trend market (trend follow favored)
        
        # === 4H HMA TREND ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean reversion long signal
        crsi_overbought = crsi[i] > 85.0  # Strong mean reversion short signal
        crsi_neutral_long = crsi[i] < 40.0  # Moderate long signal
        crsi_neutral_short = crsi[i] > 60.0  # Moderate short signal
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Choppy market + CRSI oversold (mean reversion)
        if is_choppy and crsi_oversold:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending market + trend alignment + CRSI pullback
        elif is_trending and trend_bull and hma_bull:
            if crsi_neutral_long:
                new_signal = POSITION_SIZE
        
        # Regime 3: Strong trend breakout (all aligned)
        elif trend_bull and hma_bull and crsi[i] < 50.0:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Choppy market + CRSI overbought (mean reversion)
        if is_choppy and crsi_overbought:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market + trend alignment + CRSI bounce
        elif is_trending and trend_bear and hma_bear:
            if crsi_neutral_short:
                new_signal = -POSITION_SIZE
        
        # Regime 3: Strong trend breakdown (all aligned)
        elif trend_bear and hma_bear and crsi[i] > 50.0:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if trend_bear and chop[i] < 45.0:  # Only exit if trending bear
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if trend_bull and chop[i] < 45.0:  # Only exit if trending bull
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