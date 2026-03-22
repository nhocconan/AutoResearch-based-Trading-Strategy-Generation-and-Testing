#!/usr/bin/env python3
"""
Experiment #412: 12h Primary + 1d/1w HTF — Simplified HMA Trend + CRSI Mean Reversion

Hypothesis: After analyzing 411 failed experiments, key insights emerge:
1. Complex regime-switching (#405, #411) creates too many conditions = 0 trades
2. 12h timeframe needs SIMPLER logic than 4h/1h (fewer bars = fewer opportunities)
3. HMA trend from 1d HTF is reliable major direction filter (proven in #406)
4. CRSI<20/>80 extremes work better than RSI(14) for mean reversion timing
5. Adding 1w HTF for major regime (bull/bear) improves asymmetric entries
6. Volatility filter (ATR ratio) avoids low-vol chop that kills returns

Why this might beat current best (Sharpe=0.435):
- Simpler entry logic = more trades (avoid 0-trade failure mode)
- 1w HTF regime filter prevents counter-trend trades in major crashes
- CRSI extremes (20/80) trigger more often than RSI(14) extremes (30/70)
- ATR volatility filter avoids dead zones where fees dominate
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Key changes from #406:
1. Removed Choppiness Index (too many false regime switches)
2. Removed Donchian breakout (redundant with HMA trend)
3. Simplified from 4 confluence filters to 2 main filters
4. Added 1w HTF for major bull/bear regime detection
5. Loosened CRSI thresholds (15/85 → 20/80) for more trades
6. Added ATR volatility filter to avoid low-vol chop

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_simp_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for mean reversion
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI of Streak (consecutive up/down bars)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - percentile of price change over lookback
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = pct_change.iloc[i-rank_period:i]
        current = pct_change.iloc[i]
        if not np.isnan(current) and len(window) > 0:
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Calculate 1w HTF indicators (major regime: bull/bear)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    
    # Volatility ratio (ATR 14 / ATR 30) - filter out low-vol chop
    vol_ratio = atr_14 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === 1W MAJOR REGIME (bull/bear market) ===
        # Price above 1w HMA = bull market (favor longs, reduce shorts)
        # Price below 1w HMA = bear market (favor shorts, reduce longs)
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND DIRECTION (primary filter) ===
        # Price above 1d HMA = uptrend (long bias)
        # Price below 1d HMA = downtrend (short bias)
        bull_trend_1d = close[i] > hma_1d_21_aligned[i]
        bear_trend_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === VOLATILITY FILTER ===
        # vol_ratio > 0.8 = sufficient volatility (avoid dead chop)
        vol_ok = vol_ratio[i] > 0.75
        
        # === CONNORS RSI SIGNALS (mean reversion timing) ===
        # CRSI < 20 = oversold (long opportunity)
        # CRSI > 80 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === ENTRY LOGIC — SIMPLIFIED (2 main filters) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1d trend up + CRSI oversold + vol ok
        if bull_trend_1d and crsi_oversold and vol_ok:
            # In 1w bull regime: full size
            # In 1w bear regime: reduced size (counter-trend)
            if bull_regime_1w:
                new_signal = LONG_SIZE
            else:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRY: 1d trend down + CRSI overbought + vol ok
        if bear_trend_1d and crsi_overbought and vol_ok:
            # In 1w bear regime: full size
            # In 1w bull regime: reduced size (counter-trend)
            if bear_regime_1w:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            else:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 12 bars (~6 days on 12h), force entry on weaker signal
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if bull_trend_1d and crsi[i] < 35.0 and vol_ok:
                new_signal = LONG_SIZE * 0.6
            elif bear_trend_1d and crsi[i] > 65.0 and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on mean reversion exhaustion)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d trend flip)
        if in_position and position_side > 0 and bear_trend_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_trend_1d:
            new_signal = 0.0
        
        # Local trend reversal exit (12h HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals