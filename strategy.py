#!/usr/bin/env python3
"""
Experiment #013: 1d Primary + 1w HTF | Dual Regime (Chop/Trend) + Connors RSI

Hypothesis: Daily timeframe with weekly trend filter will capture major moves while
avoiding whipsaws. Key innovation: regime-switching based on Choppiness Index.

Why this should work:
1. 1w HMA(21) establishes major trend direction (call ONCE via mtf_data)
2. Choppiness Index(14) detects regime: >61.8 = range (mean revert), <38.2 = trend
3. Connors RSI for mean reversion entries in choppy markets (75% win rate proven)
4. Donchian breakout for trending markets (proven on SOL Sharpe +0.782)
5. ATR(14) stoploss at 2.5x protects from major drawdowns
6. Funding rate contrarian filter for BTC/ETH edge through 2022 crash

Timeframe: 1d (REQUIRED for Experiment #013)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14)
Target: 20-50 trades/year on daily

Key design decisions:
- Regime detection prevents wrong strategy in wrong market
- Connors RSI (RSI2 + RSI_Streak + PercentRank) / 3 for precise mean reversion
- Funding rate z-score for contrarian edge (BTC/ETH specific)
- Minimal filters to ensure trade generation (learned from 0-trade failures)
- Discrete sizing to minimize fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_1w_hma_connors_v1"
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
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar (simplified: high - low)
    tr = high - low
    
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values
    streak_abs = np.abs(streak)
    streak_signed = np.where(streak >= 0, streak_abs, -streak_abs)
    rsi_streak = calculate_rsi(streak_signed + 100, streak_period)  # offset to avoid negatives
    
    # PercentRank: % of prior closes lower than current
    for i in range(pr_period, n):
        window = close[i-pr_period:i]
        crsi[i] = np.sum(window < close[i]) / pr_period * 100
    
    # Combine components
    for i in range(pr_period, n):
        crsi[i] = (rsi_close[i] + rsi_streak[i] + crsi[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA trend
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    
    # 1d HMA for local trend
    hma_1d_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.35)
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1w_21_aligned[i]
        htf_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        local_bullish = close[i] > hma_1d_21[i]
        local_bearish = close[i] < hma_1d_21[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        choppy_regime = chop_14[i] > 61.8  # range market
        trending_regime = chop_14[i] < 38.2  # trending market
        neutral_regime = not choppy_regime and not trending_regime
        
        # === CONNORS RSI EXTREMES (for mean reversion in chop) ===
        crsi_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 90
        
        # === BOLLINGER BAND EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === DONCHIAN BREAKOUT (for trending regime) ===
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER ===
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_bullish or local_bullish:
            current_size = BASE_SIZE
        elif htf_bearish and local_bearish:
            current_size = STRONG_SIZE
        elif htf_bearish or local_bearish:
            current_size = BASE_SIZE
        else:
            current_size = WEAK_SIZE
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: CHOPPY MARKET (mean reversion)
        if choppy_regime:
            # Long: CRSI oversold + BB lower + HTF not bearish
            if crsi_oversold and bb_oversold and not htf_bearish:
                new_signal = current_size
            # Short: CRSI overbought + BB upper + HTF not bullish
            elif crsi_overbought and bb_overbought and not htf_bullish:
                new_signal = -current_size
        
        # REGIME 2: TRENDING MARKET (breakout)
        elif trending_regime:
            # Long: Donchian breakout + HTF bullish + RSI supportive
            if donchian_long and htf_bullish and rsi_bullish:
                new_signal = current_size
            # Short: Donchian breakout + HTF bearish + RSI supportive
            elif donchian_short and htf_bearish and rsi_bearish:
                new_signal = -current_size
        
        # REGIME 3: NEUTRAL (use simpler signals)
        else:
            # Long: HTF bullish + CRSI oversold OR Donchian long
            if htf_bullish and (crsi_oversold or donchian_long):
                new_signal = current_size * 0.8
            # Short: HTF bearish + CRSI overbought OR Donchian short
            elif htf_bearish and (crsi_overbought or donchian_short):
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 45 bars (~45 days on 1d), allow weaker entry
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if htf_bullish and crsi_oversold:
                new_signal = current_size * 0.6
            elif htf_bearish and crsi_overbought:
                new_signal = -current_size * 0.6
        
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
            # Exit long if 1w trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 1w trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT (take profit on mean reversion) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
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