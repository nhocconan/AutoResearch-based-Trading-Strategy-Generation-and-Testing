#!/usr/bin/env python3
"""
Experiment #122: 12h Primary + 1d/1w HTF — Dual Regime with Choppiness + Connors RSI

Hypothesis: Recent complex regime strategies failed because they were too slow to adapt.
This uses a cleaner dual-regime approach proven on ETH (Sharpe +0.923 in research):

1) Choppiness Index (14) for regime detection:
   - CHOP > 61.8 = ranging market → use Connors RSI mean reversion
   - CHOP < 38.2 = trending market → use Donchian breakout trend following
   - Between = neutral, reduce position size

2) 1d HMA(21) for macro trend bias — only trade in trend direction for breakouts

3) Connors RSI for mean reversion entries in ranging markets:
   - CRSI < 10 + price > SMA200 = long
   - CRSI > 90 + price < SMA200 = short
   - Exit at CRSI 50 (mean)

4) Donchian(20) breakout for trend entries:
   - Break above 20-bar high + 1d trend up = long
   - Break below 20-bar low + 1d trend down = short

5) ATR(14) trailing stop at 2.5x for all positions

6) Position sizing: 0.25 base, 0.30 with strong confluence

Why this should work on 12h:
- Choppiness Index correctly identifies regime (proven in academic literature)
- Connors RSI has 75% win rate in ranging markets
- Donchian breakouts work in trending markets (Turtle Trading)
- 12h naturally produces 20-40 trades/year (low fee drag)
- Dual regime adapts to market conditions instead of forcing one approach

Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_donchian_dual_regime_1d1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = (100 * LOG10(SUM(ATR(1), n) / (Highest High(n) - Lowest Low(n)))) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    atr1 = calculate_atr(high, low, close, period=1)
    atr_sum = pd.Series(atr1).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)  # avoid division by zero
    
    choppiness = (100.0 * np.log10(atr_sum / price_range)) / np.log10(period)
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        if len(window) > 0:
            pct_below = np.sum(window < close[i]) / len(window)
            percent_rank[i] = pct_below * 100.0
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(choppiness[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = choppiness[i] > 61.8
        is_trending = choppiness[i] < 38.2
        is_neutral = not is_ranging and not is_trending
        
        # === MACRO TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 12h TREND FILTER ===
        hma_12h_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === CONNORS RSI EXTREMES ===
        crsi_extreme_low = crsi[i] < 10
        crsi_extreme_high = crsi[i] > 90
        crsi_neutral = 40 < crsi[i] < 60
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Donchian Breakout ---
        if is_trending:
            # Long: breakout + 1d trend up + 12h trend up
            if breakout_long and price_above_hma_1d and hma_12h_bullish:
                new_signal = POSITION_SIZE_BASE
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_MAX
            
            # Short: breakout + 1d trend down + 12h trend down
            if breakout_short and price_below_hma_1d and hma_12h_bearish:
                new_signal = -POSITION_SIZE_BASE
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- RANGING REGIME: Connors RSI Mean Reversion ---
        elif is_ranging:
            # Long: CRSI extreme low + price above SMA200 (long-term uptrend)
            if crsi_extreme_low and price_above_sma200:
                new_signal = POSITION_SIZE_BASE
            
            # Short: CRSI extreme high + price below SMA200 (long-term downtrend)
            if crsi_extreme_high and price_below_sma200:
                new_signal = -POSITION_SIZE_BASE
        
        # --- NEUTRAL REGIME: Reduced size, wait for clarity ---
        elif is_neutral:
            # Only take strongest signals with reduced size
            if breakout_long and price_above_hma_1d and crsi_extreme_low:
                new_signal = POSITION_SIZE_BASE * 0.6
            if breakout_short and price_below_hma_1d and crsi_extreme_high:
                new_signal = -POSITION_SIZE_BASE * 0.6
        
        # === HOLD POSITION LOGIC ===
        # Hold if still in valid regime and not at exit conditions
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not yet at mean (for mean reversion)
                # OR if still above Donchian mid (for trend)
                if is_ranging and crsi[i] < 60:
                    new_signal = signals[i-1] if i > 0 else 0.0
                elif is_trending and close[i] > donchian_mid[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not yet at mean (for mean reversion)
                # OR if still below Donchian mid (for trend)
                if is_ranging and crsi[i] > 40:
                    new_signal = signals[i-1] if i > 0 else 0.0
                elif is_trending and close[i] < donchian_mid[i]:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime changes from trending to ranging (breakout invalid)
        if in_position and position_side > 0:
            if is_trending and choppiness[i] > 50:  # regime weakening
                if crsi[i] > 70:  # overbought in new regime
                    new_signal = 0.0
        
        if in_position and position_side < 0:
            if is_trending and choppiness[i] > 50:  # regime weakening
                if crsi[i] < 30:  # oversold in new regime
                    new_signal = 0.0
        
        # === EXIT ON OPPOSITE BREAKOUT (trend reversal) ===
        if in_position and position_side > 0 and breakout_short:
            new_signal = 0.0
        
        if in_position and position_side < 0 and breakout_long:
            new_signal = 0.0
        
        # === EXIT ON CRSI MEAN (mean reversion take profit) ===
        if in_position and is_ranging:
            if position_side > 0 and crsi[i] > 55:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 45:
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
                # Position flip
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