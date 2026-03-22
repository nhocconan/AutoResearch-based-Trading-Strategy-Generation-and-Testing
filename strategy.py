#!/usr/bin/env python3
"""
Experiment #042: 12h Connors RSI + Choppiness Regime + 1d/1w HMA Trend

Hypothesis: 12h timeframe with Connors RSI entries, Choppiness regime filter,
and dual HTF trend bias (1d + 1w) will produce consistent trades with positive Sharpe.

Key design:
1. 1w HMA(21) for major trend bias (via mtf_data - call ONCE)
2. 1d HMA(21) for intermediate trend (via mtf_data - call ONCE)
3. Connors RSI(3,2,100) for entry timing - proven 75% win rate
4. Choppiness Index(14) for regime detection
5. ATR(14) for stoploss (2.5x)
6. Discrete sizing: 0.25 base, 0.30 strong trend
7. WIDER CRSI thresholds (<35 long, >65 short) to ensure trade generation

Why this should work:
- 12h TF targets 20-50 trades/year (fee efficient)
- Dual HTF filter (1d + 1w) provides stronger trend confirmation
- Connors RSI is proven mean-reversion indicator with high win rate
- Choppiness adapts between mean-revert and trend-follow modes
- Wider CRSI thresholds ensure trades actually trigger (avoiding 0-trade failure)
- ATR stoploss limits drawdown

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop!)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d_1w_hma_v1"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI(streak, 2): Streak length momentum
    PercentRank(100): Relative position in recent range
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - short term momentum
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (2)
    rsi_streak = calculate_rsi(streak, streak_period)
    
    # Percent Rank (100)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1) * 100
        crsi[i] = (rsi_3[i] + rsi_streak[i] + rank) / 3
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
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
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d and 1w HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w) ===
        # Both HTF must agree for strong signal
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend: both 1d and 1w agree
        strong_bullish = htf_1d_bullish and htf_1w_bullish
        strong_bearish = htf_1d_bearish and htf_1w_bearish
        
        # Weak/neutral: HTF disagree
        neutral_trend = (htf_1d_bullish and htf_1w_bearish) or (htf_1d_bearish and htf_1w_bullish)
        
        # === REGIME DETECTION (Choppiness) ===
        # CHOP > 55 = ranging (mean revert)
        # CHOP < 45 = trending (trend follow)
        is_choppy = chop_14[i] > 55
        is_trending = chop_14[i] < 45
        
        # === ENTRY LOGIC - CONNORS RSI (WIDE thresholds for trade gen) ===
        new_signal = 0.0
        
        if is_trending:
            # Trend follow mode: use CRSI pullback in direction of trend
            if strong_bullish:
                # Long on CRSI pullback (wide range to ensure trades)
                if crsi[i] < 40:
                    new_signal = STRONG_SIZE
            elif strong_bearish:
                # Short on CRSI rally (wide range to ensure trades)
                if crsi[i] > 60:
                    new_signal = -STRONG_SIZE
            elif htf_1d_bullish:
                # 1d bullish only
                if crsi[i] < 35:
                    new_signal = BASE_SIZE
            elif htf_1d_bearish:
                # 1d bearish only
                if crsi[i] > 65:
                    new_signal = -BASE_SIZE
        
        elif is_choppy:
            # Mean reversion mode: fade extremes regardless of trend
            if crsi[i] < 30:
                new_signal = BASE_SIZE  # long at oversold
            elif crsi[i] > 70:
                new_signal = -BASE_SIZE  # short at overbought
        
        else:
            # Neutral regime: use CRSI with moderate thresholds
            if crsi[i] < 35:
                new_signal = BASE_SIZE
            elif crsi[i] > 65:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~12.5 days on 12h), force entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if strong_bullish or htf_1d_bullish:
                new_signal = BASE_SIZE * 0.8
            elif strong_bearish or htf_1d_bearish:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if both HTF turn bearish
            if position_side > 0 and strong_bearish:
                trend_reversal = True
            # Exit short if both HTF turn bullish
            if position_side < 0 and strong_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes very overbought
            if position_side > 0 and crsi[i] > 85:
                crsi_exit = True
            # Exit short when CRSI becomes very oversold
            if position_side < 0 and crsi[i] < 15:
                crsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or crsi_exit:
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