#!/usr/bin/env python3
"""
Experiment #240: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Most 1h strategies fail due to too many trades and fee drag. This strategy
uses Choppiness Index (CHOP) to detect market regime, then applies DIFFERENT logic
per regime to reduce false signals. Key innovations:

1. CHOP > 55 = RANGE regime → Use Connors RSI mean reversion (buy CRSI<15, sell CRSI>85)
2. CHOP < 45 = TREND regime → Follow 4h HMA trend with 1h RSI pullback entries
3. CHOP 45-55 = TRANSITION → Stay flat (no trades, reduces whipsaw)
4. Session filter: Only trade 8-20 UTC (highest volume, reduces false breakouts)
5. Volume filter: Only enter if volume > 0.8x 20-bar average (confirms moves)
6. 12h HMA for macro bias alignment (only trade with 12h trend)

This should generate 30-80 trades/year (not 200+) by being selective on regime + session.
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
Stoploss: 2.5x ATR trailing (tighter for 1h timeframe)

Target: Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL), trades 30-80/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_crsi_hma_4h12h_session_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0, posinf=50.0, neginf=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): 3-period RSI on price
    RSI(streak, 2): 2-period RSI on up/down streak length
    PercentRank(100): Percentile rank of today's return over last 100 days
    """
    close_s = pd.Series(close)
    
    # RSI(3) on price
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak (use absolute values for RSI calculation)
    streak_pos = np.maximum(streak, 0)
    streak_neg = np.abs(np.minimum(streak, 0))
    
    # Simplified streak RSI: ratio of up streaks to total
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        up_sum = np.sum(np.maximum(streak[max(0, i-streak_period+1):i+1], 0))
        down_sum = np.sum(np.abs(np.minimum(streak[max(0, i-streak_period+1):i+1], 0)))
        total = up_sum + down_sum
        if total > 0:
            streak_rsi[i] = 100.0 * up_sum / total
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank: percentile of today's return over last 100 bars
    returns = close_s.pct_change()
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = returns.iloc[max(0, i-rank_period+1):i+1].dropna()
        if len(window) > 0:
            current_ret = returns.iloc[i]
            percent_rank[i] = 100.0 * np.sum(window < current_ret) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_percent_rank(series, period=100):
    """Calculate percentile rank of current value over last N periods."""
    result = np.zeros(len(series))
    for i in range(period, len(series)):
        window = series[max(0, i-period+1):i+1]
        if len(window) > 0:
            current = series[i]
            result[i] = 100.0 * np.sum(window < current) / len(window)
        else:
            result[i] = 50.0
    return result

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds, convert to hour
    hours = np.array([(prices["open_time"].iloc[i] // 3600000) % 24 for i in range(n)])
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    hma_16_1h = calculate_hma(close, 16)
    hma_48_1h = calculate_hma(close, 48)
    rsi_14_1h = calculate_rsi(close, period=14)
    atr_14_1h = calculate_atr(high, low, close, period=14)
    chop_14_1h = calculate_choppiness(high, low, close, period=14)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h HMA for trend direction (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 12h HMA for macro bias (aligned properly)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14_1h[i]) or atr_14_1h[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16_1h[i]) or np.isnan(hma_48_1h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14_1h[i]) or np.isnan(chop_14_1h[i]) or np.isnan(crsi_1h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14_1h[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        is_transition = 45.0 <= chop_value <= 55.0
        
        # === 4h TREND DIRECTION ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === 12h MACRO BIAS ===
        macro_bullish = close[i] > hma_12h_aligned[i]
        macro_bearish = close[i] < hma_12h_aligned[i]
        
        # === 1h HMA CROSSOVER ===
        hma_1h_bullish = hma_16_1h[i] > hma_48_1h[i]
        hma_1h_bearish = hma_16_1h[i] < hma_48_1h[i]
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55): Mean Reversion with CRSI ===
        if is_ranging and in_session:
            # Long: CRSI oversold + price above 12h HMA (macro bias)
            if crsi_1h[i] < 15.0 and macro_bullish and volume_confirmed:
                desired_signal = POSITION_SIZE_HALF
            
            # Short: CRSI overbought + price below 12h HMA (macro bias)
            elif crsi_1h[i] > 85.0 and macro_bearish and volume_confirmed:
                desired_signal = -POSITION_SIZE_HALF
        
        # === TRENDING REGIME (CHOP < 45): Trend Following ===
        elif is_trending and in_session:
            # Long: 4h bullish + 1h HMA bullish + RSI pullback (not overbought)
            if trend_4h_bullish and hma_1h_bullish and 35.0 <= rsi_14_1h[i] <= 60.0:
                if macro_bullish:
                    desired_signal = POSITION_SIZE_FULL
                else:
                    desired_signal = POSITION_SIZE_HALF
            
            # Short: 4h bearish + 1h HMA bearish + RSI pullback (not oversold)
            elif trend_4h_bearish and hma_1h_bearish and 40.0 <= rsi_14_1h[i] <= 65.0:
                if macro_bearish:
                    desired_signal = -POSITION_SIZE_FULL
                else:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === TRANSITION REGIME (CHOP 45-55): Stay flat ===
        # desired_signal remains 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14_1h[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14_1h[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_1h_bearish and rsi_14_1h[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_1h_bullish and rsi_14_1h[i] < 30.0:
            desired_signal = 0.0
        
        # === MACRO REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bearish and rsi_14_1h[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bullish and rsi_14_1h[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if regime still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and is_ranging and crsi_1h[i] < 50.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side > 0 and is_trending and trend_4h_bullish and rsi_14_1h[i] < 75.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and is_ranging and crsi_1h[i] > 50.0:
                desired_signal = -POSITION_SIZE_HALF
            elif position_side < 0 and is_trending and trend_4h_bearish and rsi_14_1h[i] > 25.0:
                desired_signal = -POSITION_SIZE_HALF
        
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