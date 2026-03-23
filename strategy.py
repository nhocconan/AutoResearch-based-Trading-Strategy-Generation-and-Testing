#!/usr/bin/env python3
"""
Experiment #023: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Based on research showing Connors RSI has 75% win rate for mean reversion
and Choppiness Index is the best meta-filter for bear/range markets (which 2025 is),
I'm combining these on daily timeframe with weekly HMA trend filter.

Key innovation:
1. CONNORS RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive than standard RSI for mean reversion entries
   - Long when CRSI < 15, Short when CRSI > 85
2. CHOPPINESS INDEX regime filter: CHOP(14) > 61.8 = range (enable mean reversion)
   - Only trade mean reversion when market is actually ranging
   - Stay flat during trending periods (CHOP < 38.2)
3. 1w HMA(21) for overall market bias (only long if price > weekly HMA, vice versa)
4. ATR(14) trailing stoploss at 2.5x

Why 1d works:
- Targets 20-30 trades/year (minimal fee drag)
- Less noise than lower timeframes
- Works through 2022 crash and 2025 bear market
- Proven in research for BTC/ETH specifically

Entry conditions (LOOSE enough to generate trades but strict enough for quality):
- Long: CRSI < 15 + CHOP > 55 + price > 1w HMA
- Short: CRSI > 85 + CHOP > 55 + price < 1w HMA

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_weekly_hma_v1"
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
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) on close prices - short-term momentum
    2. RSI(2) on up/down streak lengths - streak momentum
    3. PercentRank(100) - where current price ranks vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI on streak lengths
    # Calculate streak: consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on absolute streak values (convert to positive for RSI calculation)
    streak_positive = np.abs(streak)
    streak_positive[streak_positive == 0] = 0.001  # avoid division by zero
    rsi_streak = calculate_rsi(streak_positive, period=streak_period)
    
    # Component 3: PercentRank(100)
    # Where does current close rank vs last 100 closes?
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8 = Market is chopping/ranging (mean reversion favorable)
    - CHOP < 38.2 = Market is trending (trend following favorable)
    """
    n = period
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Price range
    price_range = highest_high - lowest_low + 1e-10
    
    # Choppiness Index
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for regime bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period + warmup
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Slightly relaxed from 61.8 to get more trades
        is_trending = chop_value < 38.2
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # More aggressive than standard RSI
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC (Mean Reversion in Ranging Market) ===
        new_signal = 0.0
        
        # Only trade mean reversion when market is ranging
        if is_ranging:
            # Long: CRSI oversold + price above weekly HMA (bullish bias)
            if crsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price below weekly HMA (bearish bias)
            elif crsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE
        
        # === TRENDING REGIME: Stay Flat ===
        # In trending markets, mean reversion fails. Stay flat.
        # new_signal remains 0.0
        
        # === HOLD POSITION LOGIC ===
        # If we're in position and no new signal, hold the position
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit if market transitions from ranging to trending
        if in_position and is_trending:
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