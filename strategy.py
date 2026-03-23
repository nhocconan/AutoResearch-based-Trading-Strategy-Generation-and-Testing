#!/usr/bin/env python3
"""
Experiment #229: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 19 failed 4h experiments with Fisher/KAMA/volume combos, return to
proven Choppiness Index regime detection (worked in research with ETH Sharpe +0.923).
Key innovation: DIFFERENT entry logic per regime instead of one-size-fits-all.

Regime 1 (CHOP > 61.8 = Choppy/Range): Connors RSI mean reversion
  - CRSI < 10 + price > SMA200 → Long
  - CRSI > 90 + price < SMA200 → Short
  - Expected: 60-70% win rate, quick exits

Regime 2 (CHOP < 38.2 = Trending): HMA + Donchian breakout
  - HMA16 > HMA48 + Donchian breakout + 1d bias → Long
  - HMA16 < HMA48 + Donchian breakdown + 1d bias → Short
  - Expected: Lower win rate but larger moves

Regime 3 (38.2 <= CHOP <= 61.8 = Transition): Stay flat or reduce size

1d HMA(21) macro filter: Only take longs when price > 1d HMA, shorts when < 1d HMA
This aligns with 2025 bear market reality — don't fight the macro trend.

Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
Stoploss: 2.5 * ATR(14) trailing stop
Target: 25-45 trades/year, Sharpe > 0.50 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_hma_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    n = len(close)
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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Choppy/Range market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): 3-period RSI on price
    RSI(streak, 2): 2-period RSI on up/down streak length
    PercentRank(100): Percentile rank of today's close vs last 100 closes
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on price
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0).values
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Component 3: PercentRank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    sma_200 = calculate_sma(close, 200)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start later to ensure all indicators ready (CRSI needs 100+)
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 61.8
        trending_regime = chop_14[i] < 38.2
        # transition_regime = not choppy_regime and not trending_regime
        
        # === HTF MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND DETECTION (4h HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === CONNORS RSI EXTREMES (for mean reversion) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC BY REGIME ===
        new_signal = 0.0
        
        if choppy_regime:
            # MEAN REVERSION STRATEGY
            # Long: CRSI oversold + price above SMA200 + macro bullish bias
            if crsi_oversold and close[i] > sma_200[i]:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF  # Reduced size against macro
            
            # Short: CRSI overbought + price below SMA200 + macro bearish bias
            elif crsi_overbought and close[i] < sma_200[i]:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF  # Reduced size against macro
        
        elif trending_regime:
            # TREND FOLLOWING STRATEGY
            # Long: HMA bullish + Donchian breakout + macro bullish
            if hma_bullish and breakout_long:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
                # Don't enter long against macro even in trending regime
            
            # Short: HMA bearish + Donchian breakdown + macro bearish
            elif hma_bearish and breakout_short:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
                # Don't enter short against macro even in trending regime
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if conditions still favorable
                hold_long = False
                if choppy_regime:
                    hold_long = crsi[i] < 70.0  # Hold until CRSI rises
                elif trending_regime:
                    hold_long = hma_bullish  # Hold while HMA bullish
                if hold_long:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if conditions still favorable
                hold_short = False
                if choppy_regime:
                    hold_short = crsi[i] > 30.0  # Hold until CRSI falls
                elif trending_regime:
                    hold_short = hma_bearish  # Hold while HMA bearish
                if hold_short:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if HMA crosses bearish (trending regime)
        if in_position and position_side > 0 and trending_regime and hma_bearish:
            new_signal = 0.0
        
        # Exit short if HMA crosses bullish (trending regime)
        if in_position and position_side < 0 and trending_regime and hma_bullish:
            new_signal = 0.0
        
        # Exit if macro trend strongly reverses against position
        if in_position and position_side > 0 and price_below_hma_1d and chop_14[i] < 50.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and chop_14[i] < 50.0:
            new_signal = 0.0
        
        # === CRSI EXIT (mean reversion complete) ===
        # Exit long when CRSI becomes overbought
        if in_position and position_side > 0 and choppy_regime and crsi_overbought:
            new_signal = 0.0
        
        # Exit short when CRSI becomes oversold
        if in_position and position_side < 0 and choppy_regime and crsi_oversold:
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