#!/usr/bin/env python3
"""
Experiment #604: 4h Dual-HTF Regime Adaptive (1d/1w HMA + Connors RSI + BB Width + Donchian)

Hypothesis: After 535+ failures, the key insight is that SINGLE-HTF strategies fail because
they miss major trend context. This uses DUAL-HTF (1d + 1w) for robust trend bias:

1. 1w HMA = major trend (bull/bear market)
2. 1d HMA = intermediate trend (weekly bias)
3. BB Width percentile = regime (narrow=breakout coming, wide=mean revert)
4. Connors RSI(3) = fast mean reversion entry (more signals than RSI14)
5. Donchian(20) = breakout entry when BB Width narrow + HTF aligned
6. Asymmetric sizing: 0.30 with HTF trend, 0.20 against HTF trend

Why this should beat #593 (Sharpe=-0.281):
- Dual-HTF (1d+1w) provides stronger trend confirmation than single 1d
- Connors RSI(3) generates MORE trades than RSI(14) - critical for meeting trade minimums
- BB Width percentile is more robust regime filter than CHOP
- Looser entry thresholds ensure trades generate in all market conditions
- Proper stoploss at 2*ATR with position tracking

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_htf_connors_bbwidth_donchian_regime_atr_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_close = 100 - (100 / (1 + rs))
    
    # Streak RSI: count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Percent Rank: percentage of closes in last 100 bars that are lower than current
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50
    )
    
    # CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    
    return crsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Band width as % of price
    
    return upper.values, lower.values, sma.values, width.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate percentile rank of BB Width over lookback period."""
    bb_width_s = pd.Series(bb_width)
    percentile = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50
    )
    return percentile.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_WITH_TREND = 0.30
    SIZEAgainst_TREND = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_mid[i]):
            continue
        
        if np.isnan(bb_width_pct[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === DUAL-HTF TREND BIAS ===
        # 1w HMA = major trend, 1d HMA = intermediate trend
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both align
        strong_bull = bull_1w and bull_1d
        strong_bear = bear_1w and bear_1d
        neutral = (bull_1w and bear_1d) or (bear_1w and bull_1d)
        
        # === REGIME DETECTION (BB Width Percentile) ===
        # Low percentile = narrow bands = breakout regime
        # High percentile = wide bands = mean reversion regime
        breakout_regime = bb_width_pct[i] < 30  # Bottom 30% of historical width
        meanrev_regime = bb_width_pct[i] > 70  # Top 30% of historical width
        
        # === DONCHIAN BREAKOUT (for breakout regime) ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ADX FILTER (loose for more trades) ===
        trend_strength = adx_14[i] > 15  # Loose threshold
        
        # === CONNORS RSI MEAN REVERSION (for mean reversion regime) ===
        crsi_oversold = crsi[i] < 15  # Very oversold
        crsi_overbought = crsi[i] > 85  # Very overbought
        
        # === BB EXTREMES ===
        near_bb_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        use_full_size = SIZE_WITH_TREND
        use_half_size = SIZEAgainst_TREND
        
        # MODE 1: BREAKOUT REGIME - Donchian breakout with HTF alignment
        if breakout_regime or neutral:
            # Long: Donchian breakout + HTF bullish bias
            if breakout_long and (strong_bull or bull_1d):
                new_signal = use_full_size if strong_bull else use_half_size
            
            # Short: Donchian breakout + HTF bearish bias
            elif breakout_short and (strong_bear or bear_1d):
                new_signal = -use_full_size if strong_bear else -use_half_size
        
        # MODE 2: MEAN REVERSION REGIME - Connors RSI extremes + BB touches
        if meanrev_regime or neutral:
            # Long: CRSI oversold + near BB lower
            if crsi_oversold and near_bb_lower:
                # Prefer long when HTF is bullish or neutral
                if bull_1d or neutral:
                    new_signal = use_half_size
                elif not strong_bear:
                    new_signal = use_half_size * 0.5  # Smaller against trend
            
            # Short: CRSI overbought + near BB upper
            elif crsi_overbought and near_bb_upper:
                # Prefer short when HTF is bearish or neutral
                if bear_1d or neutral:
                    new_signal = -use_half_size
                elif not strong_bull:
                    new_signal = -use_half_size * 0.5  # Smaller against trend
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and strong_bear and adx_14[i] > 20:
                trend_reversal = True
            if position_side < 0 and strong_bull and adx_14[i] > 20:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals