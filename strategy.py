#!/usr/bin/env python3
"""
Experiment #037: 1d Primary + 1w HTF — Dual Regime with Connors RSI + Donchian Breakout

Hypothesis: Daily timeframe with weekly trend bias provides optimal signal quality for 
BTC/ETH/SOL. Combining Connors RSI mean-reversion (proven 75% win rate) with Donchian 
breakouts and Choppiness Index regime detection should work across bull/bear/range markets.

Key innovations:
1. CONNORS RSI (CRSI): 3-component RSI for superior mean-reversion signals
2. CHOPPINESS INDEX: Regime switch at 50 (not 61.8/38.2 for more trades)
3. DONCHIAN BREAKOUT: 20-day high/low for trend confirmation
4. 1w HMA: Macro trend bias (only ONE HTF as per experiment rules)
5. ASYMMETRIC entries: Easier with weekly trend, harder against it

Why 1d works:
- Targets 20-50 trades/year (optimal fee efficiency per Rule 10)
- Proven in exp#026 (Sharpe=0.354), #032 (Sharpe=0.419), #033 (Sharpe=0.138)
- Less whipsaw than lower timeframes during 2022 crash

Entry conditions (LOOSE enough to generate 30+ trades):
- Long mean-revert: CRSI < 25 + CHOP > 50 + price > 1w HMA
- Short mean-revert: CRSI > 75 + CHOP > 50 + price < 1w HMA
- Long breakout: CRSI < 50 + CHOP < 50 + price breaks Donchian(20) high + 1w HMA bullish
- Short breakout: CRSI > 50 + CHOP < 50 + price breaks Donchian(20) low + 1w HMA bearish

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 3.0*ATR trailing stop (wider for daily timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_donchian_chop_regime_1w_v2"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
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
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Streak calculation (consecutive up/down days)
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if pd.isna(returns.iloc[i]):
            streak[i] = 0
            continue
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - percentile of today's return over last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period+1:i+1]
        valid_returns = window_returns.dropna()
        if len(valid_returns) > 0:
            current_return = returns.iloc[i]
            if pd.isna(current_return):
                percent_rank[i] = 50.0
            else:
                rank = np.sum(valid_returns <= current_return) / len(valid_returns)
                percent_rank[i] = rank * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
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
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=10)
    
    # Volatility ratio for additional filter
    vol_ratio = atr_14 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_30[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-1] if i >= 1 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-1] if i >= 1 else False
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Midpoint for balanced regime detection
        is_trending = chop_value < 50.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25  # Looser for more trades
        crsi_overbought = crsi[i] > 75  # Looser for more trades
        crsi_neutral_low = crsi[i] < 50
        crsi_neutral_high = crsi[i] > 50
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if i >= 1 else False
        breakout_low = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === VOLATILITY FILTER ===
        vol_expanding = vol_ratio[i] > 1.2
        vol_contracting = vol_ratio[i] < 0.9
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion ---
        if is_ranging:
            # Long: CRSI oversold + price above weekly HMA (macro support)
            if crsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price below weekly HMA (macro resistance)
            elif crsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE
            
            # Counter-trend mean reversion (weaker signal, smaller size)
            elif crsi_oversold and price_below_hma_1w and vol_expanding:
                new_signal = POSITION_SIZE * 0.67  # 2/3 size for counter-trend
            
            elif crsi_overbought and price_above_hma_1w and vol_expanding:
                new_signal = -POSITION_SIZE * 0.67
        
        # --- TRENDING REGIME: Breakout Following ---
        elif is_trending:
            # Long breakout: CRSI not overbought + breakout high + weekly bullish
            if crsi_neutral_low and breakout_high and (price_above_hma_1w or hma_1w_slope_bull):
                new_signal = POSITION_SIZE
            
            # Short breakout: CRSI not oversold + breakout low + weekly bearish
            elif crsi_neutral_high and breakout_low and (price_below_hma_1w or hma_1w_slope_bear):
                new_signal = -POSITION_SIZE
            
            # Pullback entry in trend (HMA fast crosses above/below HMA slow)
            elif crsi_neutral_low and hma_1d_fast[i] > hma_1d[i] and price_above_hma_1w:
                new_signal = POSITION_SIZE
            
            elif crsi_neutral_high and hma_1d_fast[i] < hma_1d[i] and price_below_hma_1w:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing for daily) ===
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime changes to strongly trending bearish
        if in_position and position_side > 0:
            if is_trending and hma_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        # Exit short if regime changes to strongly trending bullish
        if in_position and position_side < 0:
            if is_trending and hma_1w_slope_bull and price_above_hma_1w:
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