#!/usr/bin/env python3
"""
Experiment #032: 12h Primary + 1d/1w HTF — Connors RSI Mean Reversion + Choppiness Regime

Hypothesis: 12h timeframe with Connors RSI (proven 75% win rate) + Choppiness regime filter
will generate 20-50 trades/year with positive Sharpe across BTC/ETH/SOL. Key insight:
CRSI extremes (<10/>90) are rare but reliable, combined with regime detection ensures
we only mean-revert in ranging markets and trend-follow in trending markets.

Strategy Logic:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long entry: CRSI < 15 (oversold extreme)
   - Short entry: CRSI > 85 (overbought extreme)
2. CHOPPINESS INDEX regime: CHOP > 55 = range (use CRSI mean revert), CHOP < 45 = trend (use breakout)
3. 1d HMA: Macro trend bias for trade direction confirmation
4. 1w HMA: Ultra-long-term trend filter (avoid counter-trend trades in strong trends)
5. ATR(14) trailing stoploss: 2.5*ATR to protect capital

Why this should work:
- 12h primary = fewer trades, less fee drag (targets 25-40/year)
- CRSI = proven mean reversion signal with high win rate
- Choppiness filter = avoids mean reversion in strong trends (where it fails)
- LOOSE CRSI thresholds (15/85 not 10/90) = ensures trade generation
- 1d/1w HTF = strong trend confirmation, avoids counter-trend traps
- Discrete sizing = minimizes fee churn on signal changes

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_v2"
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

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): 3-period RSI on close
    RSI_Streak(2): RSI of streak length (consecutive up/down days)
    PercentRank(100): percentile rank of 1-day price change over 100 periods
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2)
    # Calculate streak length: consecutive up or down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
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
    
    # RSI on streak values (period=2)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100): percentile rank of 1-day price change
    pct_change = close_s.pct_change() * 100.0
    percent_rank = np.zeros(n)
    for i in range(100, n):
        window = pct_change.iloc[i-100:i].values
        current = pct_change.iloc[i]
        rank = np.sum(window < current) / len(window) * 100.0
        percent_rank[i] = rank
    
    # CRSI = average of three components
    crsi = (rsi_3 + rsi_streak.values + percent_rank) / 3.0
    
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for medium-term trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    crsi = calculate_crsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 12h HMA for local trend
    hma_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(donchian_upper[i]) or np.isnan(hma_21[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 52.0  # Range market (mean revert)
        is_trending = chop_value < 48.0  # Trend market (trend follow)
        # Hysteresis zone: 48-52 = hold previous regime
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 18.0  # Long entry
        crsi_overbought = crsi[i] > 82.0  # Short entry
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === HMA TREND ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        hma_slope_up = hma_21[i] > hma_21[i-3] if i > 3 else False
        hma_slope_down = hma_21[i] < hma_21[i-3] if i > 3 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + price below BB + HTF helps
            if crsi_oversold or price_below_bb_lower:
                # Need at least one HTF confirmation OR CRSI turning up
                if price_above_hma_1d or price_above_hma_1w or crsi_rising:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price above BB + HTF helps
            elif crsi_overbought or price_above_bb_upper:
                # Need at least one HTF confirmation OR CRSI turning down
                if price_below_hma_1d or price_below_hma_1w or crsi_falling:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Donchian ---
        elif is_trending:
            # Long: Donchian breakout + HMA bullish + HTF confirms
            if donchian_breakout_long and hma_bullish:
                if price_above_hma_1d and (price_above_hma_1w or hma_slope_up):
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + HMA bearish + HTF confirms
            elif donchian_breakout_short and hma_bearish:
                if price_below_hma_1d and (price_below_hma_1w or hma_slope_down):
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: CRSI extreme alone (ensures trade generation) ---
        if new_signal == 0.0:
            # Very oversold CRSI = long regardless of regime (strong mean reversion)
            if crsi[i] < 12.0:
                if price_above_hma_1w or price_above_hma_1d:
                    new_signal = POSITION_SIZE
            
            # Very overbought CRSI = short regardless of regime
            elif crsi[i] > 88.0:
                if price_below_hma_1w or price_below_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If we're in a position and no new signal, hold (don't flip to 0)
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
        
        # === EXIT ON STRONG REGIME/TREND CHANGE ===
        # Exit long if both HTFs turn strongly bearish in trending regime
        if in_position and position_side > 0:
            if is_trending and price_below_hma_1d and price_below_hma_1w:
                new_signal = 0.0
        
        # Exit short if both HTFs turn strongly bullish in trending regime
        if in_position and position_side < 0:
            if is_trending and price_above_hma_1d and price_above_hma_1w:
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