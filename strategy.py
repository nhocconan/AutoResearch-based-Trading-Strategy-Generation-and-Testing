#!/usr/bin/env python3
"""
Experiment #093: 1d Primary + 1w HTF — Dual Regime with Connors RSI + Donchian

Hypothesis: Previous 1d strategies failed due to either (1) too strict entry conditions 
causing 0 trades, or (2) pure trend-following that fails in bear/range markets (2025 test).

This strategy uses DUAL REGIME logic proven in research:
1) CHOPPINESS INDEX detects regime: CHOP>55 = range (mean revert), CHOP<45 = trend (breakout)
2) Range regime: Connors RSI <15 for long, >85 for short (75% win rate in research)
3) Trend regime: Donchian(20) breakout + 1w HMA trend filter
4) 1w HMA provides macro bias (only long if price>1w HMA, only short if price<1w HMA)

Why this should work:
- 1d timeframe naturally limits to 20-40 trades/year (low fee drag)
- Dual regime adapts to 2022 crash (range) and 2021 bull (trend)
- Connors RSI proven edge for mean reversion (research shows 0.8-1.5 Sharpe)
- 1w HMA prevents counter-trend trades in bear markets
- Looser entry thresholds ensure trades on ALL symbols

Position size: 0.25 base, 0.30 max with confluence
Stoploss: 3.0*ATR trailing (wider for 1d timeframe)
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_donchian_1w_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    # Percent Rank - where current price ranks in last 100 days
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100 if x.max() > x.min() else 50.0
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # Combine
    crsi = (rsi_short.fillna(50.0).values + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_ranging = chop_14[i] > 50.0  # ranging market (looser threshold for more trades)
        chop_trending = chop_14[i] < 50.0  # trending market
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        crsi_oversold = crsi[i] < 20.0  # long signal
        crsi_overbought = crsi[i] > 80.0  # short signal
        
        # === DONCHIAN BREAKOUT (trend following) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: Mean Reversion with Connors RSI ---
        if chop_ranging:
            # Long: CRSI oversold + price above 1w HMA (bullish bias)
            if crsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE_BASE
                # Boost if extremely oversold
                if crsi[i] < 15.0:
                    new_signal = POSITION_SIZE_MAX
            
            # Short: CRSI overbought + price below 1w HMA (bearish bias)
            if crsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE_BASE
                # Boost if extremely overbought
                if crsi[i] > 85.0:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- TREND REGIME: Donchian Breakout ---
        if chop_trending:
            # Long: Donchian breakout + price above 1w HMA
            if donchian_breakout_long and price_above_hma_1w:
                new_signal = POSITION_SIZE_BASE
                # Boost with strong momentum
                if close[i] > close[i-5] * 1.02:  # 2% gain in 5 days
                    new_signal = POSITION_SIZE_MAX
            
            # Short: Donchian breakout + price below 1w HMA
            if donchian_breakout_short and price_below_hma_1w:
                new_signal = -POSITION_SIZE_BASE
                # Boost with strong momentum
                if close[i] < close[i-5] * 0.98:  # 2% loss in 5 days
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought AND no trend reversal
                if crsi[i] < 75.0 and price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold AND no trend reversal
                if crsi[i] > 25.0 and price_below_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing for 1d) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE (1w HMA cross) ===
        if in_position and position_side > 0:
            if price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
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