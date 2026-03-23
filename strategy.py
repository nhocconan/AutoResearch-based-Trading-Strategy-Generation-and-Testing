#!/usr/bin/env python3
"""
Experiment #277: 1d Primary + 1w HTF — Regime-Adaptive CRSI + Choppiness

Hypothesis: Based on research showing Choppiness Index + Connors RSI achieved ETH Sharpe +0.923.
BTC/ETH fail on simple trend strategies in bear markets (2025 test period is -25%).
This strategy adapts to market regime:
- CHOP > 61.8 (choppy/range): Mean reversion via Connors RSI extremes
- CHOP < 38.2 (trending): HMA trend following with RSI pullback entries
- 1w HMA(21) for macro bias (soft filter)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.30 (conservative for daily volatility)

KEY INSIGHTS from failed experiments:
- #267, #273: 1d CRSI+Chop failed (Sharpe -0.676, -1.823) — likely over-filtered
- #270: 1h HMA+RSI failed (Sharpe -0.425) — wrong TF for this approach
- This version: SIMPLER regime logic, fewer conflicting filters, discrete signals

TARGET: 25-40 trades/year on 1d, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_atr_v2"
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
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank - percentile of today's return vs last rank_period days
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[:i+1].dropna() <= returns.iloc[i]).sum() / len(returns.iloc[:i+1].dropna()) * 100
        else:
            percent_rank[i] = 50.0
    
    # CRSI = average of three components
    crsi = (rsi_fast + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        choppiness = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    choppiness = np.nan_to_num(choppiness, nan=50.0)
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Conservative for daily volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period + indicators
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Range/mean reversion regime
        is_trending = choppiness[i] < 45.0  # Trend following regime
        
        # === MACRO BIAS (1w HMA) - SOFT FILTER ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA crossover) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY CONDITIONS
        if is_choppy:
            # Mean reversion: CRSI < 15 (extreme oversold) + price above 1w HMA (soft)
            if crsi[i] < 15.0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
        elif is_trending:
            # Trend following: HMA bullish + RSI pullback (40-55) + price above 1w HMA
            if hma_bullish and rsi_14[i] >= 40.0 and rsi_14[i] <= 55.0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
        else:
            # Neutral regime: require stronger signals
            if crsi[i] < 10.0:  # Very oversold
                desired_signal = POSITION_SIZE
            elif hma_bullish and rsi_14[i] < 45.0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
        
        # SHORT ENTRY CONDITIONS
        if is_choppy:
            # Mean reversion: CRSI > 85 (extreme overbought) + price below 1w HMA (soft)
            if crsi[i] > 85.0 and price_below_hma_1w:
                desired_signal = -POSITION_SIZE
        elif is_trending:
            # Trend following: HMA bearish + RSI pullback (45-60) + price below 1w HMA
            if hma_bearish and rsi_14[i] >= 45.0 and rsi_14[i] <= 60.0 and price_below_hma_1w:
                desired_signal = -POSITION_SIZE
        else:
            # Neutral regime: require stronger signals
            if crsi[i] > 90.0:  # Very overbought
                desired_signal = -POSITION_SIZE
            elif hma_bearish and rsi_14[i] > 55.0 and price_below_hma_1w:
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_bearish and is_trending:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish and is_trending:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (mean reversion target) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if not in strong bearish trend
                if not (hma_bearish and is_trending):
                    desired_signal = POSITION_SIZE
            elif position_side < 0:
                # Hold short if not in strong bullish trend
                if not (hma_bullish and is_trending):
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