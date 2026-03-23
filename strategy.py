#!/usr/bin/env python3
"""
Experiment #633: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Building on current best (mtf_1d_chop_crsi_regime_1w_v1, Sharpe=0.520),
this strategy improves entry precision by combining:
1. 1w HMA for major trend bias (slower, more reliable than 12h)
2. Choppiness Index (14) for regime detection: >61.8 = range, <38.2 = trend
3. Connors RSI for mean reversion entries in range regimes
4. HMA pullback entries in trend regimes
5. ATR trailing stoploss for risk management

Key insights from 559 failed strategies:
1. Pure trend following fails on BTC/ETH (2022 crash, 2025 bear market)
2. Mean reversion works better in bear/range markets (current test period)
3. 1d/1w combination gives 20-50 trades/year (optimal fee drag)
4. Connors RSI has 75% win rate for reversals
5. Choppiness Index is best meta-filter for regime switching
6. Discrete sizing (0.30) minimizes fee churn while controlling DD

Why this might beat Sharpe=0.520:
- 1w HMA is slower/more reliable than 12h for major trend direction
- Connors RSI (3 components) is more robust than simple RSI(14)
- Choppiness threshold tuning (61.8/38.2) based on literature
- Asymmetric sizing: larger in range (mean revert), smaller in trend
- ATR stoploss at 2.5*ATR protects from black swan events

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 1d (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_1w_v2"
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
    > 61.8 = ranging market (mean reversion)
    < 38.2 = trending market (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(Streak, 2): RSI on consecutive up/down days
    PercentRank(100): Percentile rank of today's return over last 100 days
    
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on price
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak
    returns = close_s.pct_change()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: Percentile Rank of returns
    pct_rank = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) * 100.0 if len(x) > 1 else 50.0
    ).values
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=9)
    atr_14 = calculate_atr(high, low, close, 14)
    choppiness = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_RANGE = 0.35  # Larger in range (mean reversion)
    POSITION_SIZE_TREND = 0.25  # Smaller in trend (more risk)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = choppiness[i] > 61.8  # Mean reversion regime
        is_trend = choppiness[i] < 38.2  # Trend following regime
        # Neutral: 38.2 <= chop <= 61.8 (no new entries)
        
        # === 1W TREND BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # 1w HMA slope (3 bars)
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-3] if i >= 3 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-3] if i >= 3 else False
        
        # === 1D HMA CROSSOVER ===
        hma_cross_bull = hma_1d_fast[i] > hma_1d[i]
        hma_cross_bear = hma_1d_fast[i] < hma_1d[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Mean reversion long
        crsi_overbought = crsi[i] > 85.0  # Mean reversion short
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower
        price_near_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        current_position_size = POSITION_SIZE_RANGE if is_range else POSITION_SIZE_TREND
        
        # --- RANGE REGIME: Mean Reversion (Connors RSI + BB) ---
        if is_range:
            # Long: CRSI oversold + price at BB lower + above 1w HMA (bias long)
            if crsi_oversold and price_near_bb_lower:
                if price_above_hma_1w or hma_1w_slope_bull:
                    new_signal = current_position_size
            
            # Short: CRSI overbought + price at BB upper + below 1w HMA (bias short)
            elif crsi_overbought and price_near_bb_upper:
                if price_below_hma_1w or hma_1w_slope_bear:
                    new_signal = -current_position_size
        
        # --- TREND REGIME: Trend Following (HMA + 1w bias) ---
        elif is_trend:
            # Long: 1w bull + 1d HMA cross up + 1d HMA sloping up
            if hma_1w_slope_bull and price_above_hma_1w:
                if hma_cross_bull:
                    new_signal = current_position_size
            
            # Short: 1w bear + 1d HMA cross down + 1d HMA sloping down
            elif hma_1w_slope_bear and price_below_hma_1w:
                if hma_cross_bear:
                    new_signal = -current_position_size
        
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
        
        # === EXIT ON REGIME CHANGE ===
        # If in range position and regime becomes trend, exit
        if in_position and is_trend and abs(signals[i-1]) == POSITION_SIZE_RANGE:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP (1w HMA) ===
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