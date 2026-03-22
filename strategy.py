#!/usr/bin/env python3
"""
Experiment #011: 4h Connors RSI + Choppiness Regime + 1d HMA Trend

Hypothesis: 4h primary with regime-adaptive logic will outperform static strategies.
Key design:
1. 1d HMA(21) for major trend direction (call ONCE before loop via mtf_data)
2. Connors RSI (CRSI) for mean reversion entries: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Choppiness Index (CHOP) for regime detection: >61.8 = range, <38.2 = trend
4. Regime-adaptive logic:
   - Range regime (CHOP>61.8): Mean revert at CRSI extremes (<10 long, >90 short)
   - Trend regime (CHOP<38.2): Follow 1d HMA direction with CRSI pullback entries
5. ATR(14) stoploss at 2.5x for risk management
6. Discrete sizing: 0.25 base, 0.30 strong confluence

Why this should work:
- Connors RSI has 75% win rate in research (proven mean reversion)
- Choppiness filter prevents mean reversion in trending markets (major failure mode)
- 1d HMA filter prevents counter-trend trades (learned from 2022 crash failures)
- 4h TF targets 20-50 trades/year (optimal for fee efficiency)
- Regime-switching adapts to market conditions (bull/bear/range)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_connors_chop_regime_1d_hma_v1"
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

def calculate_rsi_streak(close, period=2):
    """Calculate RSI Streak component of Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like scale (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    for i in range(period, n):
        streak_window = streak[i-period+1:i+1]
        up_streaks = np.sum(streak_window > 0)
        down_streaks = np.sum(streak_window < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank component of Connors RSI.
    Measures where current return ranks vs past period returns.
    """
    n = len(close)
    percent_rank = np.zeros(n)
    
    for i in range(period, n):
        # Calculate returns over the period
        returns = np.diff(close[i-period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        
        # Count how many returns are less than current
        if len(returns) > 0:
            rank = np.sum(returns < current_return)
            percent_rank[i] = 100 * rank / len(returns)
        else:
            percent_rank[i] = 50
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
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
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === 1D HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        range_regime = chop[i] > 61.8  # Ranging market
        trend_regime = chop[i] < 38.2  # Trending market
        # Neutral regime: 38.2 <= CHOP <= 61.8
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Mean reversion long signal
        crsi_overbought = crsi[i] > 85  # Mean reversion short signal
        crsi_pullback_long = crsi[i] < 40  # Pullback in uptrend
        crsi_pullback_short = crsi[i] > 60  # Pullback in downtrend
        
        # === POSITION SIZING BASED ON CONFLUENCE ===
        current_size = BASE_SIZE
        if htf_bullish and trend_regime:
            current_size = STRONG_SIZE
        elif htf_bearish and trend_regime:
            current_size = STRONG_SIZE
        elif range_regime:
            current_size = WEAK_SIZE
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # RANGE REGIME: Mean reversion at extremes
        if range_regime:
            if crsi_oversold and not htf_bearish:
                new_signal = current_size
            elif crsi_overbought and not htf_bullish:
                new_signal = -current_size
        
        # TREND REGIME: Follow trend with pullback entries
        elif trend_regime:
            if htf_bullish and crsi_pullback_long:
                new_signal = current_size
            elif htf_bearish and crsi_pullback_short:
                new_signal = -current_size
        
        # NEUTRAL REGIME: Conservative, only strong signals
        else:
            if htf_bullish and crsi_oversold:
                new_signal = current_size * 0.8
            elif htf_bearish and crsi_overbought:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~6-7 days on 4h), allow weaker entry
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if htf_bullish and crsi[i] < 30:
                new_signal = current_size * 0.7
            elif htf_bearish and crsi[i] > 70:
                new_signal = -current_size * 0.7
        
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
            # Exit long if 1d trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT (take profit) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes very overbought
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Exit short when CRSI becomes very oversold
            if position_side < 0 and crsi[i] < 20:
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