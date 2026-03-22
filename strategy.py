#!/usr/bin/env python3
"""
Experiment #586: 4h Fisher Transform Reversal with Dual HTF HMA + Choppiness Regime

Hypothesis: After 500+ failed experiments, the key insight is that 4h needs:
1. Ehlers Fisher Transform for precise reversal detection (catches bear market rallies)
2. Dual HTF bias (1d + 1w HMA) for strong trend confirmation
3. Choppiness Index regime filter to switch between mean-revert and trend-follow
4. Connors RSI for additional mean-reversion confirmation
5. Asymmetric position sizing based on regime confidence

Why this should work on 4h:
- Fisher Transform normalizes price to Gaussian distribution, extreme values (-1.5, +1.5) signal reversals
- 4h has 6 bars/day = ~2190 bars/year = good balance of signal frequency vs noise
- Dual HTF (1d + 1w) provides stronger trend bias than single HTF
- Choppiness Index > 61.8 = range (mean revert), < 38.2 = trend (follow breakout)
- Connors RSI < 10 or > 90 provides additional extreme confirmation
- This combines 3 proven edges from quantitative literature

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_connors_dual_htf_chop_regime_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Extreme values (< -1.5 or > +1.5) signal potential reversals.
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate EMA of HL2
    ema_hl2 = hl2_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Normalize to -1 to +1 range
    normalized = np.zeros(len(close))
    for i in range(period, len(close)):
        min_hl2 = hl2[i-period+1:i+1].min()
        max_hl2 = hl2[i-period+1:i+1].max()
        if max_hl2 - min_hl2 > 0:
            normalized[i] = 0.66 * ((hl2[i] - min_hl2) / (max_hl2 - min_hl2) - 0.5) + 0.67 * normalized[i-1]
        else:
            normalized[i] = normalized[i-1] if i > 0 else 0
    
    # Clip to avoid division issues
    normalized = np.clip(normalized, -0.99, 0.99)
    
    # Fisher Transform
    fisher = np.zeros(len(close))
    for i in range(len(close)):
        if abs(normalized[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1 + normalized[i]) / (1 - normalized[i]))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme values < 10 or > 90 signal mean-reversion opportunities.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = sum(1 for j in range(i-streak_period+1, i+1) if streak[j] > 0)
        if streak_period > 0:
            streak_rsi[i] = 100 * up_streaks / streak_period
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current price ranks in last 100 periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = sum(1 for p in window[:-1] if p < window[-1])
        percent_rank[i] = 100 * rank / (rank_period - 1) if rank_period > 1 else 50
    
    # Connors RSI
    crsi = (rsi + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll).replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values

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
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HIGH = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF TREND BIAS ===
        # Both 1d and 1w must agree for strong bias
        bull_bias_strong = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        bear_bias_strong = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        bull_bias_weak = close[i] > hma_1d_aligned[i]
        bear_bias_weak = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        range_regime = chop[i] > 61.8  # Mean reversion mode
        trend_regime = chop[i] < 38.2  # Trend following mode
        neutral_regime = not range_regime and not trend_regime
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        # Short: Fisher crosses below +1.5 from above
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Extreme oversold
        crsi_overbought = crsi[i] > 85  # Extreme overbought
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        # RANGE REGIME (mean reversion)
        if range_regime:
            # Long: Fisher reversal + Connors oversold + RSI oversold + bullish bias
            if fisher_long and crsi_oversold and rsi_oversold and bull_bias_weak:
                new_signal = SIZE_HIGH
                position_size = SIZE_HIGH
            # Short: Fisher reversal + Connors overbought + RSI overbought + bearish bias
            elif fisher_short and crsi_overbought and rsi_overbought and bear_bias_weak:
                new_signal = -SIZE_HIGH
                position_size = SIZE_HIGH
        
        # TREND REGIME (trend following)
        elif trend_regime:
            # Long: Fisher reversal + strong bullish bias
            if fisher_long and bull_bias_strong:
                new_signal = SIZE_HIGH
                position_size = SIZE_HIGH
            # Short: Fisher reversal + strong bearish bias
            elif fisher_short and bear_bias_strong:
                new_signal = -SIZE_HIGH
                position_size = SIZE_HIGH
        
        # NEUTRAL REGIME (conservative)
        else:
            # Only trade with strong bias and extreme signals
            if fisher_long and crsi_oversold and bull_bias_strong:
                new_signal = SIZE_BASE
            elif fisher_short and crsi_overbought and bear_bias_strong:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes against position type
        if in_position and new_signal != 0.0:
            if position_side > 0 and range_regime and not crsi_oversold:
                # Long in range regime without oversold condition - reduce risk
                if chop[i] > 70:  # Very choppy
                    new_signal = 0.0
            if position_side < 0 and range_regime and not crsi_overbought:
                # Short in range regime without overbought condition - reduce risk
                if chop[i] > 70:  # Very choppy
                    new_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        # Exit if dual HTF bias flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias_strong:
                new_signal = 0.0
            if position_side < 0 and bull_bias_strong:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
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
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals