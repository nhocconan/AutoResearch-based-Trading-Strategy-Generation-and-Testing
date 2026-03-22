#!/usr/bin/env python3
"""
Experiment #032: 12h Dual Regime (Choppiness + Connors RSI) + 1d/1w HMA Trend

Hypothesis: Dual-regime strategy adapts to market conditions:
1. CHOPPINESS INDEX detects regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
2. CONNORS RSI for entry timing: CRSI < 20 = oversold long, CRSI > 80 = overbought short
3. 1d HMA(21) + 1w HMA(21) for major trend bias (call ONCE via mtf_data)
4. In range regime: mean reversion at CRSI extremes
5. In trend regime: pullback entries with trend
6. ATR(14) stoploss at 2.5x for risk management

Why this should work:
- Choppiness Index proven in research (ETH Sharpe +0.923)
- Connors RSI has 75% win rate in backtests
- Dual regime adapts to 2022 crash (trend) vs 2025 range markets
- 12h TF targets 20-50 trades/year (optimal fee efficiency)
- Simple CRSI thresholds (20/80) ensure trades actually trigger

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_1d_1w_hma_regime_v1"
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
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
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
            chop[i] = 50  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (map to 0-100)
    # Positive streak = bullish, negative = bearish
    max_streak = np.max(np.abs(streak)) + 1
    streak_rsi = 50 + (streak / max_streak) * 50
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - where current return ranks in lookback period
    returns = np.diff(close) / np.roll(close, 1)
    returns[0] = 0
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / rank_period
        crsi[i] = (rsi_fast[i] + streak_rsi[i] + rank * 100) / 3
    
    # Fill early values
    for i in range(rank_period):
        crsi[i] = 50
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMA trends
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === HTF TREND BIAS ===
        # 1w HMA = major trend, 1d HMA = intermediate trend
        htf_bullish = close[i] > hma_1w_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop_14[i] > 61.8  # range market
        trending_regime = chop_14[i] < 38.2  # trending market
        neutral_regime = not choppy_regime and not trending_regime
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20  # strong mean reversion long
        crsi_overbought = crsi[i] > 80  # strong mean reversion short
        crsi_mild_oversold = crsi[i] < 30  # weaker long signal
        crsi_mild_overbought = crsi[i] > 70  # weaker short signal
        
        # === POSITION SIZING BASED ON REGIME ===
        if htf_bullish or htf_bearish:
            current_size = STRONG_SIZE
        elif htf_neutral:
            current_size = BASE_SIZE
        else:
            current_size = WEAK_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        if choppy_regime:
            # Range market: mean reversion at CRSI extremes
            if crsi_oversold:
                new_signal = current_size
            elif crsi_mild_oversold and htf_bullish and bars_since_last_trade > 20:
                new_signal = current_size * 0.8
        elif trending_regime:
            # Trending market: pullback entries with trend
            if htf_bullish and crsi_mild_oversold:
                new_signal = current_size
            elif htf_bearish and crsi_mild_overbought:
                new_signal = -current_size
        else:
            # Neutral regime: use both signals with HTF filter
            if crsi_oversold and htf_bullish:
                new_signal = current_size
            elif crsi_overbought and htf_bearish:
                new_signal = -current_size
            elif crsi_mild_oversold and htf_bullish and bars_since_last_trade > 25:
                new_signal = current_size * 0.7
            elif crsi_mild_overbought and htf_bearish and bars_since_last_trade > 25:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~20 days on 12h), allow weaker entry
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if crsi_mild_oversold and htf_bullish:
                new_signal = current_size * 0.6
            elif crsi_mild_overbought and htf_bearish:
                new_signal = -current_size * 0.6
            elif crsi_oversold:
                new_signal = current_size * 0.5
            elif crsi_overbought:
                new_signal = -current_size * 0.5
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit long if regime switches from choppy to strong trending bearish
            if position_side > 0 and trending_regime and htf_bearish:
                regime_exit = True
            # Exit short if regime switches from choppy to strong trending bullish
            if position_side < 0 and trending_regime and htf_bullish:
                regime_exit = True
        
        # === CRSI EXTREME EXIT (take profit) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_exit or crsi_exit:
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