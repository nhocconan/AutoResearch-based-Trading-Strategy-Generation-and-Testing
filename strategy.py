#!/usr/bin/env python3
"""
Experiment #283: 1d Primary + 1w HTF — Regime-Adaptive CRSI/Donchian

Hypothesis: Daily timeframe with weekly trend filter + regime detection will work better than lower timeframes.
- 1w HMA(21) for MACRO trend bias (only trade in direction of weekly trend)
- 1d Choppiness Index(14) for regime: CHOP>61.8 = range (use CRSI mean reversion), CHOP<38.2 = trend (use Donchian breakout)
- Connors RSI for mean reversion entries in choppy regimes (CRSI<30 long, CRSI>70 short)
- Donchian(20) breakout for trend entries in trending regimes
- ATR(14) 3x trailing stoploss
- Position size: 0.30 (conservative for daily volatility)

Why this might work:
1. Higher TF (1d) = fewer trades, less fee drag (~20-50 trades/year target)
2. Regime switching adapts to market conditions (bear/range vs bull/trend)
3. 1w filter prevents counter-trend trades in strong macro trends
4. CRSI proven on ETH (Sharpe +0.923 in research), Donchian proven on SOL (Sharpe +0.782)
5. Conservative sizing (0.30) limits drawdown during 2022 crash

TARGET: Sharpe > 0.5 on ALL symbols, DD < -40%, 20-50 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

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
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak_abs[i] * 25)
        else:
            streak_rsi[i] = max(0, 50 - streak_abs[i] * 25)
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    crsi = (rsi_3 + streak_rsi + percent_rank.values) / 3
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    tr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        choppiness = 100 * np.log10(tr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1w HMA for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Conservative for daily
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range/mean reversion regime
        is_trending = chop[i] < 38.2  # Trend regime
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        exit_triggered = False
        
        # --- STOPLOSS CHECK (3 * ATR trailing) ---
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                exit_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                exit_triggered = True
        
        # --- REGIME CHANGE EXIT ---
        if in_position and not exit_triggered:
            if position_side > 0 and is_trending and price_below_hma_1w:
                exit_triggered = True
            if position_side < 0 and is_trending and price_above_hma_1w:
                exit_triggered = True
        
        # --- CRSI EXTREME EXIT (take profit in choppy regime) ---
        if in_position and not exit_triggered and is_choppy:
            if position_side > 0 and crsi[i] > 70.0:
                exit_triggered = True
            if position_side < 0 and crsi[i] < 30.0:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        else:
            # --- CHOPPY REGIME: CRSI Mean Reversion ---
            if is_choppy:
                # Long: CRSI < 30 (oversold) + price above weekly HMA
                if crsi[i] < 30.0 and price_above_hma_1w:
                    desired_signal = POSITION_SIZE
                # Short: CRSI > 70 (overbought) + price below weekly HMA
                elif crsi[i] > 70.0 and price_below_hma_1w:
                    desired_signal = -POSITION_SIZE
            
            # --- TRENDING REGIME: Donchian Breakout ---
            elif is_trending:
                # Long: Price breaks Donchian upper + price above weekly HMA
                if close[i] > donchian_upper[i-1] and price_above_hma_1w:
                    desired_signal = POSITION_SIZE
                # Short: Price breaks Donchian lower + price below weekly HMA
                elif close[i] < donchian_lower[i-1] and price_below_hma_1w:
                    desired_signal = -POSITION_SIZE
            
            # --- HOLD LOGIC ---
            if in_position and desired_signal == 0.0:
                if position_side > 0:
                    # Hold long if still in valid regime
                    if (is_choppy and price_above_hma_1w and crsi[i] < 70.0) or \
                       (is_trending and price_above_hma_1w):
                        desired_signal = POSITION_SIZE
                elif position_side < 0:
                    # Hold short if still in valid regime
                    if (is_choppy and price_below_hma_1w and crsi[i] > 30.0) or \
                       (is_trending and price_below_hma_1w):
                        desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals