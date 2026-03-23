#!/usr/bin/env python3
"""
Experiment #332: 12h Primary + 1d/1w HTF — Triple Regime with Funding Rate

Hypothesis: 12h timeframe provides optimal balance between trade frequency (20-50/year)
and signal quality. Previous 12h failures (#322, #326) had 0 trades from over-filtering.

KEY IMPROVEMENTS:
1. Use 1d HMA for BIAS only (not hard entry filter) — allows more trades
2. Use 1w HMA for ultra-long-term regime (bull/bear market detection)
3. Choppiness Index (14) for regime: >61.8=range, <38.2=trend
4. Connors RSI for mean reversion in chop (thresholds 12/88 for more triggers)
5. Donchian(20) breakout for trend following (NO additional filters)
6. Funding rate contrarian signal when available (z-score < -2 long, > +2 short)
7. ATR(14) trailing stoploss at 2.5*ATR
8. Position size: 0.28 (discrete: 0.0, ±0.28)

REGIME LOGIC:
- 1w HMA: Price > 1w HMA = bull market (favor longs), Price < 1w HMA = bear (favor shorts)
- 1d HMA: Price > 1d HMA = short-term bullish bias
- Choppiness: >61.8 = range (mean revert), <38.2 = trend (breakout)

ENTRY CONDITIONS (LOOSE to ensure trades):
- Range: CRSI < 12 (long) or > 88 (short) + aligned with 1w bias
- Trend: Donchian breakout + aligned with 1d bias

TARGET: Sharpe > 0.7 on ALL symbols, 25-40 trades/year, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_triple_regime_crsi_donchian_funding_v1"
timeframe = "12h"
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
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0)
    
    # RSI Streak (2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    gain_streak = streak_s.diff().clip(lower=0)
    loss_streak = (-streak_s.diff()).clip(lower=0)
    avg_gain_streak = gain_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss_streak = loss_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + rs_streak))
    streak_rsi = streak_rsi.fillna(50.0)
    streak_rsi = np.where(delta > 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    return crsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_funding_zscore(prices, lookback=30):
    """
    Calculate funding rate z-score if available.
    Returns array of z-scores, or None if funding data not available.
    """
    try:
        import os
        symbol = os.environ.get('SYMBOL', 'BTCUSDT')
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            if len(funding_df) >= lookback:
                funding_rates = funding_df['funding_rate'].values
                mean_funding = pd.Series(funding_rates).rolling(window=lookback, min_periods=lookback).mean().values
                std_funding = pd.Series(funding_rates).rolling(window=lookback, min_periods=lookback).std().values
                with np.errstate(divide='ignore', invalid='ignore'):
                    zscore = (funding_rates - mean_funding) / (std_funding + 1e-10)
                return zscore
    except:
        pass
    return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for short-term bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Try to load funding rate z-score
    funding_zscore = calculate_funding_zscore(prices, lookback=30)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO REGIME (1w HMA) — LONG-TERM BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === SHORT-TERM BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Standard thresholds: >61.8=range, <38.2=trend
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        # Between 38.2-61.8 = neutral, default to trend logic
        
        # === FUNDING RATE SIGNAL (if available) ===
        funding_signal = 0.0
        if funding_zscore is not None and i < len(funding_zscore) and not np.isnan(funding_zscore[i]):
            if funding_zscore[i] < -2.0:
                funding_signal = 1.0  # Extreme negative funding = long
            elif funding_zscore[i] > 2.0:
                funding_signal = -1.0  # Extreme positive funding = short
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Connors RSI Mean Reversion
            # LOOSE thresholds: CRSI<12 long, >88 short (more triggers than 15/85)
            if crsi[i] < 12.0:
                # Long signal — check 1w bias for strength
                if price_above_hma_1w:
                    desired_signal = POSITION_SIZE  # Bull + oversold = strong long
                else:
                    desired_signal = POSITION_SIZE * 0.6  # Bear + oversold = weak long
            elif crsi[i] > 88.0:
                # Short signal — check 1w bias for strength
                if price_below_hma_1w:
                    desired_signal = -POSITION_SIZE  # Bear + overbought = strong short
                else:
                    desired_signal = -POSITION_SIZE * 0.6  # Bull + overbought = weak short
        
        elif is_trending:
            # TREND REGIME: Donchian Breakout
            # LONG: Price breaks Donchian upper
            if close[i] > donchian_upper[i-1]:
                if price_above_hma_1d:
                    desired_signal = POSITION_SIZE  # Bullish bias + breakout = strong long
                else:
                    desired_signal = POSITION_SIZE * 0.6  # Bearish bias + breakout = weak long
            # SHORT: Price breaks Donchian lower
            elif close[i] < donchian_lower[i-1]:
                if price_below_hma_1d:
                    desired_signal = -POSITION_SIZE  # Bearish bias + breakdown = strong short
                else:
                    desired_signal = -POSITION_SIZE * 0.6  # Bullish bias + breakdown = weak short
        
        else:
            # NEUTRAL REGIME (38.2-61.8): Use funding rate if available, else Donchian
            if funding_signal != 0.0:
                # Funding contrarian signal
                if funding_signal > 0 and price_above_hma_1w:
                    desired_signal = POSITION_SIZE * 0.7
                elif funding_signal < 0 and price_below_hma_1w:
                    desired_signal = -POSITION_SIZE * 0.7
            else:
                # Default to Donchian breakout
                if close[i] > donchian_upper[i-1]:
                    desired_signal = POSITION_SIZE * 0.7
                elif close[i] < donchian_lower[i-1]:
                    desired_signal = -POSITION_SIZE * 0.7
        
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
        
        # === CRSI EXTREME EXIT (take profit in range regime) ===
        if is_choppy and in_position and position_side > 0 and crsi[i] > 75.0:
            desired_signal = 0.0
        
        if is_choppy and in_position and position_side < 0 and crsi[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                desired_signal = POSITION_SIZE
            elif position_side < 0:
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