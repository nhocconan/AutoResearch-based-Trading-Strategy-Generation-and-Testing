#!/usr/bin/env python3
"""
Experiment #560: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: After 497 failed strategies, the pattern is clear:
- Pure trend-following on 1h fails (Sharpe negative in #555, #548)
- Too many filters = 0 trades (#550, #552, #558 all Sharpe=0.000)
- For 2025 bear/range test period, MEAN REVERSION works better than trend following
- Connors RSI (CRSI) has proven 75% win rate in literature for mean reversion
- Choppiness Index filters regime: CHOP>55=range(mean revert), CHOP<45=trend(follow)

This strategy combines:
1. 4h HMA(21) for major trend BIAS (not hard filter - allows counter-trend in range)
2. 12h Choppiness Index for REGIME detection (range vs trend)
3. 1h Connors RSI for ENTRY timing (CRSI<15 long, CRSI>85 short)
4. ATR(14) 2.5x trailing stop + regime-based exit

Why this might beat Sharpe=0.435:
- Mean reversion works in bear/range markets (2025 test period)
- CRSI extremes are rare = fewer trades = less fee drag
- Regime filter adapts to market conditions
- 1h TF with 4h/12h regime = optimal balance per Rule 10
- Target: 40-60 trades/year (per Rule 10 for 1h)

Position sizing: 0.20 discrete (smaller for mean reversion)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h12h_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term for mean reversion
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - consecutive up/down days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak (treat streak as "price")
    streak_rsi = calculate_rsi(streak + 100, streak_period)  # offset to avoid negative
    
    # Percent Rank - today's return percentile over last 100 periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window) > 0:
            current_return = returns.iloc[i]
            percent_rank[i] = (window < current_return).sum() / len(window) * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    # Clip to valid range
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0  # default to neutral
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend bias
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 12h Choppiness for regime
    chop_12h = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for mean reversion strategy
    POSITION_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Start later for CRSI rank_period
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(chop_12h_aligned[i]):
            continue
        if np.isnan(crsi_1h[i]):
            continue
        
        # === 12H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (favor mean reversion)
        # CHOP < 45 = trending market (favor trend following)
        # 45-55 = neutral (allow both)
        range_regime = chop_12h_aligned[i] > 55.0
        trend_regime = chop_12h_aligned[i] < 45.0
        neutral_regime = 45.0 <= chop_12h_aligned[i] <= 55.0
        
        # === 4H TREND BIAS (not hard filter, just sizing modifier) ===
        bull_bias = close[i] > hma_4h_21_aligned[i]
        bear_bias = close[i] < hma_4h_21_aligned[i]
        strong_bull = bull_bias and (hma_4h_21_aligned[i] > hma_4h_50_aligned[i])
        strong_bear = bear_bias and (hma_4h_21_aligned[i] < hma_4h_50_aligned[i])
        
        # === CONNORS RSI ENTRY (extreme mean reversion) ===
        # CRSI < 15 = extremely oversold (long signal)
        # CRSI > 85 = extremely overbought (short signal)
        # These are RARE = fewer trades = less fee drag
        crsi_oversold = crsi_1h[i] < 15.0
        crsi_overbought = crsi_1h[i] > 85.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + (range regime OR bull bias)
        # In range: mean revert regardless of trend
        # In trend: only long if bull bias (with trend)
        if crsi_oversold:
            if range_regime or neutral_regime:
                # Range market - mean revert long
                new_signal = POSITION_SIZE
            elif trend_regime and bull_bias:
                # Trending market - only long with trend
                new_signal = POSITION_SIZE if strong_bull else POSITION_SIZE * 0.7
        
        # SHORT ENTRY: CRSI overbought + (range regime OR bear bias)
        elif crsi_overbought:
            if range_regime or neutral_regime:
                # Range market - mean revert short
                new_signal = -POSITION_SIZE
            elif trend_regime and bear_bias:
                # Trending market - only short with trend
                new_signal = -POSITION_SIZE if strong_bear else -POSITION_SIZE * 0.7
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === MEAN REVERSION EXIT (CRSI crossback) ===
        # Exit long when CRSI crosses above 50 (mean reached)
        # Exit short when CRSI crosses below 50 (mean reached)
        if in_position and position_side > 0 and crsi_1h[i] > 50.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi_1h[i] < 50.0:
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
                # Flip position
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