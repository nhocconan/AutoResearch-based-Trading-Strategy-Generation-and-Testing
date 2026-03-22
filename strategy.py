#!/usr/bin/env python3
"""
Experiment #592: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime

Hypothesis: After analyzing 500+ failed strategies, clear patterns emerge:
- Connors RSI (CRSI) has 75% win rate in literature for mean reversion entries
- Choppiness Index (CHOP) is best regime filter (CHOP>61.8=range, <38.2=trend)
- 12h timeframe reduces noise vs 4h/1h while generating 30-50 trades/year
- 1d/1w HTF provides major trend bias to filter counter-trend disasters
- Dual regime: mean reversion in chop, trend follow in trend

Why this might beat Sharpe=0.520:
- CRSI < 20 / > 80 (wider than <10/>90) ensures sufficient trades
- CHOP regime switch prevents trend strategies in chop (major loss source)
- 12h TF optimal balance: fewer false signals than 4h, more trades than 1d
- 1w HMA for ultra-long-term bias (filters 2022 crash scenarios)
- Asymmetric sizing: 0.30 when HTF confirms, 0.20 when neutral

Entry Logic:
- CRSI < 20 = oversold (long opportunity in chop regime)
- CRSI > 80 = overbought (short opportunity in chop regime)
- CHOP > 61.8 = chop regime (use mean reversion)
- CHOP < 38.2 = trend regime (use trend following)
- 1d HMA(21) + 1w HMA(21) for trend bias
- ATR(14) 2.5x trailing stop on all positions

Position sizing: 0.20-0.30 discrete (Rule 4, max 0.40)
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_v1"
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
    Calculate Connors RSI (CRSI) - proven mean reversion signal.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Literature shows 75% win rate for CRSI < 20 (long) and CRSI > 80 (short).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak duration
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(np.abs(streak) + 1e-10, streak_period)
    
    # Component 3: Percent rank of price change
    pct_change = close_s.pct_change()
    pct_rank = pct_change.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ) * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + pct_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs for trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.20
    CONFIRMED_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        # === REGIME DETECTION ===
        is_chop_regime = chop_14[i] > 61.8
        is_trend_regime = chop_14[i] < 38.2
        
        # === HTF TREND BIAS ===
        # 1d bias
        bull_bias_1d = close[i] > hma_1d_21_aligned[i]
        bear_bias_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d slope
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # 1w bias (ultra long-term)
        bull_bias_1w = close[i] > hma_1w_21_aligned[i]
        bear_bias_1w = close[i] < hma_1w_21_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Mean reversion in chop regime (CRSI extremes) - PRIMARY SIGNAL
        if is_chop_regime:
            # Long: CRSI < 20 (oversold) + 1d bull bias OR neutral 1w
            if crsi[i] < 20.0:
                if bull_bias_1d or (not bear_bias_1w):
                    size = CONFIRMED_SIZE if (bull_bias_1d and hma_1d_slope_bull) else BASE_SIZE
                    new_signal = size
            
            # Short: CRSI > 80 (overbought) + 1d bear bias OR neutral 1w
            elif crsi[i] > 80.0:
                if bear_bias_1d or (not bull_bias_1w):
                    size = CONFIRMED_SIZE if (bear_bias_1d and hma_1d_slope_bear) else BASE_SIZE
                    new_signal = -size
        
        # Trend following in trend regime - SECONDARY SIGNAL
        elif is_trend_regime:
            # Long: 1d bull + (1w bull OR 1d slope bull)
            if bull_bias_1d and (bull_bias_1w or hma_1d_slope_bull):
                new_signal = CONFIRMED_SIZE
            
            # Short: 1d bear + (1w bear OR 1d slope bear)
            elif bear_bias_1d and (bear_bias_1w or hma_1d_slope_bear):
                new_signal = -CONFIRMED_SIZE
        
        # Neutral regime: only extreme CRSI
        else:
            if crsi[i] < 15.0 and bull_bias_1d:
                new_signal = BASE_SIZE
            elif crsi[i] > 85.0 and bear_bias_1d:
                new_signal = -BASE_SIZE
        
        # === HOLD POSITION ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
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
        
        # === EXIT ON BIAS FLIP ===
        if in_position and position_side > 0:
            if bear_bias_1d and hma_1d_slope_bear:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if bull_bias_1d and hma_1d_slope_bull:
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