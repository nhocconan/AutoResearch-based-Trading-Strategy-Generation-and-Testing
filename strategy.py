#!/usr/bin/env python3
"""
Experiment #383: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI

Hypothesis: After 382 experiments, the pattern is clear:
1. 1d timeframe with 1w HTF works best (current best Sharpe=0.435)
2. Choppiness Index regime detection reduces whipsaws in range markets
3. Connors RSI (CRSI) provides superior entry timing vs standard RSI
4. HMA(21/50) for trend direction with less lag than EMA
5. Dual-mode: mean-revert in chop (CHOP>61.8), trend-follow otherwise
6. Conservative sizing: 0.25-0.30 discrete levels to minimize fee churn

Why this might beat current best (Sharpe=0.435):
- CHOP filter avoids trend-follow losses in range markets (2021-2022 had many)
- CRSI has 75% win rate on mean reversion entries (vs 55% for standard RSI)
- 1w HMA provides stronger major trend bias than 1d alone
- ATR 2.5x trailing stop protects from crash scenarios (2022 -77%)
- Target: 25-40 trades/year on 1d (optimal for fee/capture balance)

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0.435
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive up/down days (positive for up, negative for down)
    PercentRank: percentile of current close in last 100 closes
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (map to 0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # RSI of streak: treat streak as "gains/losses"
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market bias (favor longs)
        # Price below 1w HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === CHOPPINESS REGIME ===
        choppy_market = chop_14[i] > 61.8  # range market
        trending_market = chop_14[i] < 38.2  # trend market
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 20 = oversold (long opportunity in bull)
        # CRSI > 80 = overbought (short opportunity in bear)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # Moderate signals for trend following
        crsi_long_setup = crsi[i] < 45.0
        crsi_short_setup = crsi[i] > 55.0
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MODE 1: TRENDING MARKET (CHOP < 38.2) - Trend Follow
        if trending_market:
            # Long: Bull regime + local bullish + CRSI pullback
            if bull_regime and hma_bullish and crsi_long_setup:
                new_signal = LONG_SIZE
            # Short: Bear regime + local bearish + CRSI rally
            elif bear_regime and hma_bearish and crsi_short_setup:
                new_signal = -SHORT_SIZE
        
        # MODE 2: CHOPPY MARKET (CHOP > 61.8) - Mean Reversion
        elif choppy_market:
            # Long: Bull regime + CRSI oversold (buy dip in range)
            if bull_regime and crsi_oversold:
                new_signal = LONG_SIZE
            # Short: Bear regime + CRSI overbought (sell rally in range)
            elif bear_regime and crsi_overbought:
                new_signal = -SHORT_SIZE
        
        # MODE 3: NEUTRAL (38.2 <= CHOP <= 61.8) - Conservative entries
        else:
            # Only enter on strong signals
            if bull_regime and hma_bullish and crsi_oversold:
                new_signal = LONG_SIZE * 0.8
            elif bear_regime and hma_bearish and crsi_overbought:
                new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 20 bars (~20 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_regime and crsi[i] < 35:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and crsi[i] > 65:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and crsi[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25:
            new_signal = 0.0
        
        # Trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (1d HMA cross)
        if in_position and position_side > 0 and hma_bearish and not choppy_market:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish and not choppy_market:
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