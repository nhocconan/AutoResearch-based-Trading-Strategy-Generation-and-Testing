#!/usr/bin/env python3
"""
Experiment #103: 6h Primary + 1d/1w HTF — Fisher Transform + Vol Spike Reversion + Regime

Hypothesis: 6h timeframe sits between 4h and 12h - captures multi-day swings without excessive noise.
After 6h experiment #100 failed (Donchian+HMA+RSI+Chop, Sharpe=-0.549), I'm trying a DIFFERENT approach:

Key insights from research:
1. Vol spike reversion (ATR(7)/ATR(30) > 2.0 + price < BB) captures "vol crush" after panic - proven edge
2. Fisher Transform catches reversals in bear rallies (period=9, cross above -1.5 = long)
3. Connors RSI has 75% win rate for mean reversion entries
4. BTC/ETH need mean-reversion bias, not pure trend following (2022 crash proved this)

Strategy design:
- Timeframe: 6h (28 candles/week, 30-60 trades/year target)
- HTF: 1d HMA(50) for major trend bias, 1w HMA(21) for secular bias
- Core signal: Vol spike reversion (ATR ratio > 2.0) + Fisher reversal
- Regime: Choppiness Index to filter (only mean-revert when CHOP > 50)
- Entry: CRSI < 15 (oversold) + vol spike + Fisher cross + HTF not strongly against
- Position size: 0.27 (27% of capital, conservative for 6h swings)
- Stoploss: 2.5x ATR trailing

Why this might work where #100 failed:
- #100 used Donchian breakouts (trend-following) which failed in 2022 bear
- This uses vol spike REVERSION (counter-trend after panic) which works in crashes
- Fisher Transform specifically designed for reversal detection in non-Gaussian markets
- Looser CRSI threshold (< 15 not < 10) ensures enough trades generate

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_volspike_crsi_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures ranging vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian-like distribution
    Catches reversals in non-Gaussian markets (bear rallies)
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate median price
        median = (high[i] + low[i]) / 2.0
        
        # Calculate highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_hl = highest_high - lowest_low
        if range_hl < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * ((median - lowest_low) / range_hl - 0.5)
        normalized = np.clip(normalized, -0.99, 0.99)  # prevent division issues
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = 0.0
    
    return fisher, fisher_prev

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, percent_rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    Long: CRSI < 10-15, Short: CRSI > 85-90
    """
    n = len(close)
    if n < percent_rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - measures consecutive up/down days
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_values = streak[i-streak_period+1:i+1]
        positive_streaks = np.sum(streak_values > 0)
        if streak_period > 0:
            streak_rsi[i] = 100.0 * positive_streaks / streak_period
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100) - where does current return rank vs last 100 bars
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    for i in range(percent_rank_period, n):
        current_return = returns[i]
        past_returns = returns[i-percent_rank_period+1:i]
        count_lower = np.sum(past_returns < current_return)
        percent_rank[i] = 100.0 * count_lower / (percent_rank_period - 1)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(percent_rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Bollinger Bands for mean reversion levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for secular bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, percent_rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    
    signals = np.zeros(n)
    SIZE = 0.27  # 27% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === VOL SPIKE DETECTION (ATR ratio) ===
        # ATR(7)/ATR(30) > 2.0 = vol spike (panic/extreme move)
        vol_spike = (atr_7[i] / atr_30[i]) > 2.0 if atr_30[i] > 1e-10 else False
        
        # === FISHER TRANSFORM REVERSAL ===
        # Long: Fisher crosses above -1.5 from below
        # Short: Fisher crosses below +1.5 from above
        fisher_long_cross = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_cross = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # loose threshold for trades
        crsi_overbought = crsi[i] > 85.0
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.002  # at or below lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.998  # at or above upper band
        
        # === CHOPPINESS REGIME ===
        # CHOP > 50 = ranging (favor mean reversion)
        # CHOP < 50 = trending (favor trend following)
        is_choppy = chop[i] > 50.0
        
        # === DESIRED SIGNAL (Vol Spike Reversion + Fisher + CRSI) ===
        desired_signal = 0.0
        
        # LONG ENTRY: Vol spike + CRSI oversold + Fisher reversal + HTF not strongly bear
        if vol_spike and crsi_oversold and fisher_long_cross:
            if not htf_1w_bear:  # weekly not strongly bearish
                desired_signal = SIZE
            elif htf_1d_bull:  # or daily is bullish
                desired_signal = SIZE * 0.7
        
        # Alternative LONG: Near BB lower + CRSI oversold + choppy regime
        elif near_bb_lower and crsi_oversold and is_choppy:
            if not htf_1w_bear:
                desired_signal = SIZE * 0.8
        
        # SHORT ENTRY: Vol spike + CRSI overbought + Fisher reversal + HTF not strongly bull
        elif vol_spike and crsi_overbought and fisher_short_cross:
            if not htf_1w_bull:  # weekly not strongly bullish
                desired_signal = -SIZE
            elif htf_1d_bear:  # or daily is bearish
                desired_signal = -SIZE * 0.7
        
        # Alternative SHORT: Near BB upper + CRSI overbought + choppy regime
        elif near_bb_upper and crsi_overbought and is_choppy:
            if not htf_1w_bull:
                desired_signal = -SIZE * 0.8
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals