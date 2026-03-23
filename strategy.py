#!/usr/bin/env python3
"""
Experiment #017: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + Weekly Trend

Hypothesis: Based on research showing Connors RSI achieves 75% win rate with proper 
SMA200 filter, and Choppiness Index successfully switches between mean-reversion 
(ranging) and trend-following regimes. Using 1d primary timeframe targets 20-50 
trades/year naturally (fee-efficient). 1w HMA provides long-term trend bias to 
avoid counter-trend trades in strong trends.

Key innovation:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI(14)
   - Captures short-term oversold/overbought extremes
2. Choppiness Index regime detection
   - CHOP > 61.8 = RANGING → Use CRSI mean reversion
   - CHOP < 38.2 = TRENDING → Use breakout + trend follow
3. 1w HMA for long-term bias (only long if 1w HMA bullish)
4. ATR trailing stop (3.0 * ATR) for risk management

Why 1d works:
- Naturally targets 20-50 trades/year (Rule 10 compliant)
- Less noise than lower TFs, avoids whipsaw
- Proven in research for crypto perpetual futures
- Works through 2022 crash and 2025 bear market

Entry conditions (LOOSE enough to generate trades):
- Long in range: CRSI < 15 + CHOP > 61.8 + price > 1w HMA
- Short in range: CRSI > 85 + CHOP > 61.8 + price < 1w HMA
- Long in trend: Price > 1w HMA + CRSI < 40 + CHOP < 38.2
- Short in trend: Price < 1w HMA + CRSI > 60 + CHOP < 38.2

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 3.0*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
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
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: Percentage of past returns lower than current return
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=range(n), dtype=float)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current_return = returns.iloc[i]
        if pd.isna(current_return):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current_return).sum()
            percent_rank.iloc[i] = 100.0 * rank / rank_period
    
    percent_rank = percent_rank.fillna(50.0).values
    
    # Combine components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for long-term trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # 1d HMA for shorter-term trend
    hma_1d = calculate_hma(close, period=21)
    
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(hma_1d[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND ===
        price_above_hma_1d = close[i] > hma_1d[i]
        price_below_hma_1d = close[i] < hma_1d[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_moderate_low = crsi[i] < 40
        crsi_moderate_high = crsi[i] > 60
        
        # === DUAL REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + price above 1w HMA (bullish long-term bias)
            if crsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price below 1w HMA (bearish long-term bias)
            elif crsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following ---
        elif is_trending:
            # Long: Price above 1w HMA + CRSI moderate low (pullback entry)
            if price_above_hma_1w and crsi_moderate_low:
                new_signal = POSITION_SIZE
            
            # Short: Price below 1w HMA + CRSI moderate high (rally entry)
            elif price_below_hma_1w and crsi_moderate_high:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if we're already in one and no exit signal
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
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
        # Exit long if regime changes to trending bearish
        if in_position and position_side > 0:
            if is_trending and price_below_hma_1w:
                new_signal = 0.0
        
        # Exit short if regime changes to trending bullish
        if in_position and position_side < 0:
            if is_trending and price_above_hma_1w:
                new_signal = 0.0
        
        # === EXIT ON CRSI REVERSAL ===
        # Exit long if CRSI becomes overbought
        if in_position and position_side > 0:
            if crsi_overbought:
                new_signal = 0.0
        
        # Exit short if CRSI becomes oversold
        if in_position and position_side < 0:
            if crsi_oversold:
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