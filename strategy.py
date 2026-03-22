#!/usr/bin/env python3
"""
Experiment #525: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After 470+ failed strategies, the key insight is that lower TF (1h) needs
VERY STRICT entry filters to avoid fee drag. This strategy uses:

1. 1d HMA(21) for MAJOR trend direction (only trade with HTF trend)
2. 4h Choppiness Index for REGIME detection (range vs trending)
3. 1h Connors RSI for ENTRY timing (proven 75% win rate mean reversion)
4. Session filter (8-20 UTC) for liquidity
5. Volume filter (>0.8x 20-bar avg) for confirmation
6. ATR(14) 2.5x trailing stop for risk management

Why this might work:
- Connors RSI combines RSI(3) + RSI_Streak(2) + PercentRank(100) — catches oversold/overbought
- Choppiness Index distinguishes range (mean revert) vs trend (follow) regimes
- 1d trend filter prevents counter-trend trades (major failure mode in 2022)
- Session filter avoids low-liquidity hours (reduces slippage)
- Small position size (0.20) for lower TF to control drawdown

Target: 30-80 trades/year (strict entry = 3-7 per month per symbol)
Position sizing: 0.20 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_chop_session_4h1d_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) — proven mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    
    Research shows 75% win rate with proper filters.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi_raw = pd.Series(streak).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rsi = 100.0 / (1.0 + np.exp(-streak_rsi_raw.values / 3.0 + 5.0))  # Sigmoid transform
    
    # Component 3: Percentile Rank of close over lookback
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        percent_rank[i] = np.sum(window < current) / rank_period * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) — regime detection.
    
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Range-bound market (mean reversion)
    - CHOP < 38.2: Trending market (trend following)
    - 38.2 - 61.8: Transition zone
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_series = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    for i in range(period, n):
        atr_sum = atr_series[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high > lowest_low:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0  # Neutral if no range
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h Choppiness Index for regime detection
    chop_4h = calculate_choppiness_index(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Connors RSI for entry timing
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume MA for filter
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: extract hour from open_time (UTC)
    # open_time is in milliseconds
    hours = pd.to_datetime(open_time, unit='ms').hour
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for lower TF)
    POSITION_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(chop_4h_aligned[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H REGIME (Choppiness Index) ===
        chop_value = chop_4h_aligned[i]
        range_regime = chop_value > 55.0  # Range-bound (mean revert)
        trend_regime = chop_value < 45.0  # Trending (follow trend)
        
        # === SESSION FILTER (8-20 UTC for liquidity) ===
        session_ok = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME FILTER (>0.8x average) ===
        volume_ok = volume[i] > 0.8 * volume_ma20[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold for long
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought for short
        
        # === ENTRY LOGIC — STRICT CONFLUENCE (3+ filters) ===
        new_signal = 0.0
        
        # LONG ENTRIES (require: bull regime + session + volume + CRSI extreme)
        if bull_regime and session_ok and volume_ok:
            # Condition 1: Range regime + CRSI oversold (mean reversion in uptrend)
            if range_regime and crsi_oversold:
                new_signal = POSITION_SIZE
            # Condition 2: Trend regime + CRSI oversold + HMA slope (pullback in trend)
            elif trend_regime and crsi_oversold and hma_slope_bull:
                new_signal = POSITION_SIZE
            # Condition 3: Very extreme CRSI (<10) + bull regime (strong signal)
            elif crsi[i] < 10.0 and bull_regime:
                new_signal = POSITION_SIZE
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0 and bear_regime and session_ok and volume_ok:
            # Condition 1: Range regime + CRSI overbought (mean reversion in downtrend)
            if range_regime and crsi_overbought:
                new_signal = -POSITION_SIZE
            # Condition 2: Trend regime + CRSI overbought + HMA slope (bounce in downtrend)
            elif trend_regime and crsi_overbought and hma_slope_bear:
                new_signal = -POSITION_SIZE
            # Condition 3: Very extreme CRSI (>90) + bear regime (strong signal)
            elif crsi[i] > 90.0 and bear_regime:
                new_signal = -POSITION_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip or CRSI extreme) ===
        # Exit long on regime flip to bear or extreme CRSI
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif crsi[i] > 90.0:  # Extreme overbought exit
                new_signal = 0.0
        
        # Exit short on regime flip to bull or extreme CRSI
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif crsi[i] < 10.0:  # Extreme oversold exit
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